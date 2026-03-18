/**********************************************************************
  dampingF.c:

     dampingF.c is a subroutine to calculate a damping function which
     is used for calculation of nonlocal projector matrices.

  Log of dampingF.c:

     16/Oct/2005  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"

static void inv(int n, double a[3][3], double ia[3][3]);

double dampingF(double rcut, double r)
{
  int i,j,k;
  double c0,c1,c2,c3,c4,c5;
  double buf,r01,r02,r03,r04,r05;
  double r0,r3,r4,r5,f;
  double a[3][3];
  double ia[3][3];

  buf = 1.0;

  r01 = -buf; 
  r02 = r01*r01;
  r03 = r02*r01;
  r04 = r03*r01;
  r05 = r04*r01;

  a[0][0] =     r03; a[0][1] =     r04; a[0][2] =     r05;
  a[1][0] = 3.0*r02; a[1][1] = 4.0*r03; a[1][2] = 5.0*r04;
  a[2][0] = 6.0*r01; a[2][1] =12.0*r02; a[2][2] =20.0*r03;

  inv(2,a,ia);

  c0 = 0.0;
  c1 = 0.0;
  c2 = 0.0;

  c3 = ia[0][0];
  c4 = ia[1][0];
  c5 = ia[2][0];

  r0 = r - rcut;
  r3 = r0*r0*r0;
  r4 = r0*r3;
  r5 = r0*r4;

  if (rcut<r)
    f = 0.0;
  else if (r<(rcut-buf))
    f = 1.0;
  else 
    f = c3*r3 + c4*r4 + c5*r5;

  return f;
}




void inv(int n, double a[3][3], double ia[3][3])
{
  int i,j,k;
  double w;
  double x[3],y[3];
  double da[3][3];

  /***************************************************
                     start calc.
  ****************************************************/

  if (n==-1){
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                       LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }
    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

}



