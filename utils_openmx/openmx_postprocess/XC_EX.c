/**********************************************************************
  XC_EX.c:

     XC_EX.c is a subroutine to calculate local exchange energy
     density and potential.

     This routine was written by T.Ozaki, based on the original fortran 
     code provided by the SIESTA group through their website.
     Thanks to them.

  Log of XC_EX.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

#define ZERO          0.00
#define ONE           1.00
#define PFIVE         0.50
#define OPF           1.50
#define C014          0.0140
#define TRD           0.333333333333333
#define ALP           0.666666666666666
#define FTRD          1.333333333333333
#define NINETH        0.111111111111111
#define TFTM          0.519842099789746       /*  2**FTRD-2      */
#define A0            0.521061761197848       /* (4/(9*PI))**TRD */
#define den_min       1.0e-14
#define den_min_half  0.5e-14
#define COE1          0.6203504908994   /* pow(3.0/4.0/PI,1.0/3.0); */
#define COE2          0.610887057710857 /* 3.0*ALP/(2.0*PI*A0)      */

void XC_EX(int NSP, double DS0, double DS[2], double EX[1], double VX[2])
{
  double D0,D1,D,Z,FZ,FZP;
  double RS,VXP,EXP,VXF,EXF;

  if (NSP==2){
    D = DS[0] + DS[1];
    if (D<=den_min){
      RS = COE1*pow(den_min,-1.0/3.0);
      D0 = den_min_half;
      D1 = den_min_half; 
      D = den_min;
    }
    else{
      RS = COE1*pow(D,-TRD);
    }
    Z = (D0 - D1)/D;
    FZ = (pow(1.0 + Z,FTRD) + pow(1.0 - Z,FTRD) - 2.0)/TFTM;
    FZP = FTRD*(pow(1.0 + Z,TRD) - pow(1.0 - Z,TRD))/TFTM;
  }
    
  else{ 
    if (DS0<=den_min){
      D = den_min;
      RS = COE1*pow(den_min,-1.0/3.0);
    }
    else{
      D = DS0;
      RS = COE1*pow(D,-TRD);
    }
    Z = ZERO;
    FZ = ZERO;
    FZP = ZERO;
  }

  VXP = -COE2/RS;
  EXP = 0.750*VXP;
  VXF = NINETH*VXP;
  EXF = NINETH*EXP;

  if (NSP==2){
    VX[0] = VXP + FZ*(VXF - VXP) + (1.0 - Z)*FZP*(EXF - EXP);
    VX[1] = VXP + FZ*(VXF - VXP) - (1.0 + Z)*FZP*(EXF - EXP);
    EX[0] = EXP + FZ*(EXF - EXP);
  }
  else{
    VX[0] = VXP;
    EX[0] = EXP;
  } 
}
