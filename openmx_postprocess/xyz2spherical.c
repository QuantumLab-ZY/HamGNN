/**********************************************************************
  xyz2spherical.c:

     xyz2spherical.c is a subrutine to transform xyz coordinates to
     sherical coordinates

  Log of xyz2spherical.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

void xyz2spherical(double x, double y, double z,
                   double xo, double yo, double zo,
                   double S_coordinate[3])
{
  double dx,dy,dz,r,r1,theta,phi,dum,dum1,Min_r;

  Min_r = 10e-15;

  dx = x - xo;
  dy = y - yo;
  dz = z - zo;

  dum = dx*dx + dy*dy; 
  r = sqrt(dum + dz*dz);
  r1 = sqrt(dum);

  if (Min_r<=r){

    if (r<fabs(dz))
      dum1 = sgn(dz)*1.0;
    else
      dum1 = dz/r;

    theta = acos(dum1);

    if (Min_r<=r1){

      if (0.0<=dx){

        if (r1<fabs(dy))
          dum1 = sgn(dy)*1.0;
        else
          dum1 = dy/r1;        
  
        phi = asin(dum1);
      }
      else{

        if (r1<fabs(dy))
          dum1 = sgn(dy)*1.0;
        else
          dum1 = dy/r1;        

        phi = PI - asin(dum1);
      }

    }
    else{
      phi = 0.0;
    }
  }
  else{
    theta = 0.5*PI;
    phi = 0.0;
  }

  S_coordinate[0] = r;
  S_coordinate[1] = theta;
  S_coordinate[2] = phi;
}

