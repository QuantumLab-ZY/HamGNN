/**********************************************************************
  KumoF.c:

     KumoF.c is a subroutine to calculate an interpolated value of yv
     at x using the "Kumogata" interpolation.

  Log of KumoF.c:

     26/Feb./2012  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

#ifdef MAX 
#undef MAX
#endif
#define MAX(a,b) (((a)>(b))?  (a):(b)) 

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (((a)<(b))?  (a):(b))

double KumoF(int N, double x, double *xv, double *rv, double *yv)
{

  if (x<xv[0]){

    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df,r;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;

    r = exp(x); 
    
    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;      
    return a*r*r + b;

  }

  else{

    int i;
    double t,dt,y;
    double xmin,xmax;

    xmin = xv[0];
    xmax = xv[N-1];
    x = MIN(x,xmax);
    x = MAX(x,xmin); 
    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    i = floor(t); 
    dt = t - (double)i; 

    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
		  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt 
		 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1]; 
  }

}



