/**********************************************************************
  ReLU_inverse.c:

     ReLU_inverse.c is a subroutine to calculate the inverse of matrix
     constructed by real number using LU factorization.

  Log of ReLU_inverse.c:

     22/Nov/2001  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"

void ReLU_inverse(int n, double **a, double **ia)
{

  static int i,j,k;
  static double w;
  static double *x,*y;
  static double **da;

  /****************************************************
    allocation of arrays:

    static double x[List_YOUSO[7]];
    static double y[List_YOUSO[7]];
    static double da[List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  da = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  /***************************************************
                     start calc.
  ****************************************************/

  if (n==-1){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
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

  /****************************************************
    freeing of arrays:

    static double x[List_YOUSO[7]];
    static double y[List_YOUSO[7]];
    static double da[List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[7]; i++){
    free(da[i]);
  }
  free(da);

}



