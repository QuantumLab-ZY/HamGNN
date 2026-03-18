/**********************************************************************
  Smoothing_Func.c:

     Smoothing_Func.c is a subroutine to calculate a smoothing function
     which is used in the Krylov subspace method to smear out the
     contribution of Hamiltonian and overlap matrices near the surface
     of the truncated cluster. 

  Log of Smoothing_Func.c:

     12/July/2006  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Smoothing_Func(double rcut,double r1)
{
  double sf,a,r;
  double c0,c1,c2,c3;

  a = 1.0;

  if (rcut<r1){
    sf = 0.0;
  }
  else if (r1<=(rcut-a)){
    sf = 1.0;
  }
  else{
    c0 = 1.0; 
    c1 = 0.0;
    c2 = -3.0/(a*a);
    c3 = 2.0/(a*a*a);
    r = r1 - (rcut - a);
    sf = c3*r*r*r + c2*r*r + c0;
  }

  return sf;
}
