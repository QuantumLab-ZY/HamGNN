/**********************************************************************
  LU_inverse.c:

     LU_inverse.c is a subroutine to calculate the inverse of matrix
     constructed by complex numbers using LU factorization.

  Log of LU_inverse.c:

     22/Nov/2001  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"

void LU_inverse(int n, dcomplex **a)
{
  int i,j,k;
  dcomplex w,sum;
  dcomplex *x,*y;
  dcomplex **ia;
  dcomplex **b;
  dcomplex **da;

  /****************************************************
    allocation of arrays:

    static dcomplex x[List_YOUSO[7]];
    static dcomplex y[List_YOUSO[7]];
    static dcomplex ia[List_YOUSO[7]][List_YOUSO[7]];
    static dcomplex b[List_YOUSO[7]][List_YOUSO[7]];
    static dcomplex da[List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  x = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  y = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);

  ia = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ia[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  }

  b = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    b[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  }

  da = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    da[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  }

  /***************************************************
                     start calc.
  ****************************************************/

  if (n==-1){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	a[i][j] = Complex(0.0,0.0);
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
      w = RCdiv(1.0,a[k][k]);
      for (i=k+1; i<=n; i++){
	a[i][k] = Cmul(w,a[i][k]);
	for (j=k+1; j<=n; j++){
	  a[i][j] = Csub(a[i][j], Cmul(a[i][k],a[k][j]));
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = Complex(1.0,0.0);
	else
	  y[i] = Complex(0.0,0.0);
	for (j=0; j<=i-1; j++){
	  y[i] = Csub(y[i],Cmul(a[i][j],y[j]));
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = Csub(x[i],Cmul(a[i][j],x[j]));
	}
	x[i] = Cdiv(x[i],a[i][i]);
	ia[i][k] = x[i];
      }
    }
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	sum = Complex(0.0,0.0);
	for (k=0; k<=n; k++){
	  sum = Cadd(sum,Cmul(da[i][k],ia[k][j]));
	}
	b[i][j] = sum;
      }
    }
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	a[i][j] = ia[i][j];
      }
    }
  }

  /****************************************************
    freeing of arrays:

    static dcomplex x[List_YOUSO[7]];
    static dcomplex y[List_YOUSO[7]];
    static dcomplex ia[List_YOUSO[7]][List_YOUSO[7]];
    static dcomplex b[List_YOUSO[7]][List_YOUSO[7]];
    static dcomplex da[List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ia[i]);
  }
  free(ia);

  for (i=0; i<List_YOUSO[7]; i++){
    free(b[i]);
  }
  free(b);

  for (i=0; i<List_YOUSO[7]; i++){
    free(da[i]);
  }
  free(da);

}






