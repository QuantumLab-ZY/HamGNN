/**********************************************************************
  Spherical_Bessel.c:

     Spherical_Bessel.c is a subroutine to calculate the spherical 
     Bessel functions and its derivative from 0 to lmax

  Log of Spherical_Bessel.c:

     08/Nov/2005  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
  
#define xmin  0.0
#define asize_lmax 30


void Spherical_Bessel( double x, int lmax, double *sb, double *dsb ) 
{
  int m,n,nmax,p;
  double tsb[asize_lmax+10];
  double invx,vsb0,vsb1,vsb2,vsbi;
  double j0,j1,j0p,j1p,sf,tmp,si,co,ix,ix2;

  if (x<0.0){
    printf("minus x is invalid for Spherical_Bessel\n");
    exit(0);    
  } 

  /* find an appropriate nmax */

  nmax = lmax + (int)(1.5*x) + 10;

  if (nmax<30) nmax = 30; 

  if (asize_lmax<(lmax+1)){
    printf("asize_lmax should be larger than %d in Spherical_Bessel.c\n",lmax+1);
    exit(0);
  }

  /* if x is larger than xmin */

  if ( xmin < x ){

    double *tmp_array; 

    invx = 1.0/x;

    /* allocation of array */
    tmp_array = (double*)malloc(sizeof(double)*(nmax+1));

    for ( n=0; n<nmax; n++){
      tmp_array[n] = (2.0*(double)n + 1.0)*invx;
    }

    /* initial values */

    vsb0 = 0.0;
    vsb1 = 1.0e-14;
    
    /* downward recurrence from nmax-2 to lmax+2 */

    for ( n=nmax-1; (lmax+2)<n; n-- ){
      
      vsb2 = tmp_array[n]*vsb1 - vsb0;

      if (0.0<(vsb2-1.0e+250)){
        tmp = 1.0/vsb2;
        vsb1 *= tmp;
        vsb2 = 1.0;
      }

      vsbi = vsb0;
      vsb0 = vsb1;
      vsb1 = vsb2;
    }

    /* downward recurrence from lmax+1 to 0 */

    n = lmax + 3;
    tsb[n-1] = vsb1;
    tsb[n  ] = vsb0;
    tsb[n+1] = vsbi;

    tmp = tsb[n-1];        
    tsb[n-1] /= tmp;
    tsb[n  ] /= tmp;

    for ( n=lmax+2; 0<n; n-- ){

      tsb[n-1] = tmp_array[n]*tsb[n] - tsb[n+1];

      if (1.0e+250<tsb[n-1]){
        tmp = tsb[n-1];
        for (m=n-1; m<=lmax+1; m++){
          tsb[m] /= tmp;
        }
      }
    }

    /* normalization */

    si = sin(x);
    co = cos(x);
    ix = 1.0/x;
    ix2 = ix*ix;
    j0 = si*ix;
    j1 = si*ix*ix - co*ix;

    if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
    else                           sf = j1/tsb[1];

    /* tsb to sb */

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = tsb[n]*sf;
    }

    /* derivative of sb */

    dsb[0] = co*ix - si*ix*ix;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }

    /* freeing of array */
    free(tmp_array);
  } 

  /* if x is smaller than xmin */

  else {

    /* sb */

    for ( n=0; n<=lmax; n++ ){
      sb[n] = 0.0;
    }
    sb[0] = 1.0;

    /* derivative of sb */

    dsb[0] = 0.0;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }
}

