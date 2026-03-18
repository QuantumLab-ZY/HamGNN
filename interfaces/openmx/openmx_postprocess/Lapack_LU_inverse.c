#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "tran_prototypes.h"
#include "lapack_prototypes.h"


/* typedef struct { double r,i; } dcomplex; */

/*
#ifdef MUL_TEST
#undef MUL_TEST   
#endif
*/


int LU_Zinverse_Kino(int n, dcomplex *A);
int LU_Zinverse_Ozaki(int n, dcomplex *A);
static dcomplex MyComplex(double re, double im);
static dcomplex MyCadd(dcomplex a, dcomplex b);
static dcomplex MyCmul(dcomplex a, dcomplex b);
static dcomplex MyCdiv(dcomplex a, dcomplex b);
static dcomplex MyCsub(dcomplex a, dcomplex b);
static dcomplex MyRCdiv(double x, dcomplex a);


int Lapack_LU_Zinverse(int n, dcomplex *A)
{

  LU_Zinverse_Kino(n,A);

  return 0;
}
  









int LU_Zinverse_Kino(int n, dcomplex *A)
#define C_ref(i,j) C[n*(j)+i]
{
    static char *thisprogram="Lapack_LU_inverse";
    int *ipiv;
    dcomplex *work;
    int lwork;
    int info;

#ifdef MUL_TEST
    int i,j;
    dcomplex *B,*C;
    dcomplex alpha,beta;
    double eps=1.0e-13;
    alpha.r=1.0; alpha.i=0.0;
    beta.r=0.0; beta.i=0.0;
    B=(dcomplex*)malloc(sizeof(dcomplex)*n*n);
    C=(dcomplex*)malloc(sizeof(dcomplex)*n*n);
    for (i=0;i<n*n;i++) B[i]=A[i];
#endif

    /* L*U factorization */

    ipiv = (int*) malloc(sizeof(int)*n);

    F77_NAME(zgetrf,ZGETRF)(&n,&n,A,&n,ipiv,&info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
      return 1;
    }

    /* inverse L*U factorization */

    if (0){

      work = (dcomplex*)malloc(sizeof(dcomplex)*n);
      lwork=-1;
      F77_NAME(zgetri,ZGETRI)(&n,A,&n,ipiv,work,&lwork,&info); /* query size of lwork */

      if ( info==0 ) {
	lwork=(int)work[0].r;
	free(work);
	work = (dcomplex*)malloc(sizeof(dcomplex)*(lwork));
      }
      else {
	lwork = n;
      }

    }
    else {
      lwork = 4*n;
      work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);
    }

    F77_NAME(zgetri,ZGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
      return 1;
    }

    free(work); free(ipiv);

#ifdef MUL_TEST

    printf("A*B test n=%d\n",n);
    F77_NAME(zgemm,ZGEMM)("N","N", &n,&n,&n, &alpha, A,&n, B, &n, &beta, C, &n);
    for (j=0; j<n; j++) {
      for (i=0; i<n; i++) {
	if (i==j && ( fabs(C_ref(i,j).r-1.0) > eps ||  fabs(C_ref(i,j).i) > eps )) {
	  printf("error in %d %d C=%le %le\n",i,j,C_ref(i,j).r,C_ref(i,j).i);
	  exit(0);
	}
	if (i!=j && ( fabs(C_ref(i,j).r) > eps ||  fabs(C_ref(i,j).i) > eps ) ) {
	  printf("error in %d %d C=%le %le\n",i,j,C_ref(i,j).r,C_ref(i,j).i);
	  exit(0);
	}
      }
    }
    free(C);
    free(B);

#endif

    return 0;
}






int LU_Zinverse_Ozaki(int n, dcomplex *A)
{
  static int i,j,k,vecsize; 
  static dcomplex w,sum;
  static dcomplex *x,*y;
  static dcomplex **ia;
  static dcomplex **a;
  static dcomplex **b;
  static dcomplex **da;

  vecsize = n + 4;

  /****************************************************
    allocation of arrays:

    static dcomplex x[vecsize];
    static dcomplex y[vecsize];
    static dcomplex ia[vecsize][vecsize];
    static dcomplex a[vecsize][vecsize];
    static dcomplex b[vecsize][vecsize];
    static dcomplex da[vecsize][vecsize];
  ****************************************************/

  x = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);
  y = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);

  ia = (dcomplex**)malloc(sizeof(dcomplex*)*vecsize);
  for (i=0; i<vecsize; i++){
    ia[i] = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);
  }

  a = (dcomplex**)malloc(sizeof(dcomplex*)*vecsize);
  for (i=0; i<vecsize; i++){
    a[i] = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);
  }

  b = (dcomplex**)malloc(sizeof(dcomplex*)*vecsize);
  for (i=0; i<vecsize; i++){
    b[i] = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);
  }

  da = (dcomplex**)malloc(sizeof(dcomplex*)*vecsize);
  for (i=0; i<vecsize; i++){
    da[i] = (dcomplex*)malloc(sizeof(dcomplex)*vecsize);
  }

  /***************************************************
     A -> a
  ****************************************************/

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      a[i][j].r = A[i*n+j].r;
      a[i][j].i = A[i*n+j].i;
    }
  }

  /***************************************************
                     start calc.
  ****************************************************/

  if (n==-1){
    for (i=0; i<vecsize; i++){
      for (j=0; j<vecsize; j++){
	a[i][j].r = 0.0;
	a[i][j].i = 0.0;
      }
    }
  }
  else{
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                       LU factorization
    ****************************************************/

    for (k=0; k<(n-1); k++){
      w = MyRCdiv(1.0,a[k][k]);
      for (i=k+1; i<n; i++){
	a[i][k] = MyCmul(w,a[i][k]);
	for (j=k+1; j<n; j++){
	  a[i][j] = MyCsub(a[i][j], MyCmul(a[i][k],a[k][j]));
	}
      }
    }

    for (k=0; k<n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<n; i++){
	if (i==k)
	  y[i] = MyComplex(1.0,0.0);
	else
	  y[i] = MyComplex(0.0,0.0);
	for (j=0; j<=(i-1); j++){
	  y[i] = MyCsub(y[i], MyCmul(a[i][j],y[j]));
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=(n-1); 0<=i; i--){
	x[i] = y[i];
	for (j=(n-1); (i+1)<=j; j--){
	  x[i] = MyCsub(x[i],MyCmul(a[i][j],x[j]));
	}
	x[i] = MyCdiv(x[i],a[i][i]);
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
	sum = MyComplex(0.0,0.0);
	for (k=0; k<n; k++){
	  sum = MyCadd(sum,MyCmul(da[i][k],ia[k][j]));
	}
	b[i][j] = sum;
      }
    }

    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
	a[i][j] = ia[i][j];
      }
    }
  }

  /****************************************************
    a -> A
  ****************************************************/

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      A[i*n+j] = a[i][j];
    }
  }
  
  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(x);
  free(y);

  for (i=0; i<vecsize; i++){
    free(ia[i]);
  }
  free(ia);

  for (i=0; i<vecsize; i++){
    free(a[i]);
  }
  free(a);

  for (i=0; i<vecsize; i++){
    free(b[i]);
  }
  free(b);

  for (i=0; i<vecsize; i++){
    free(da[i]);
  }
  free(da);

  return 0;
}






dcomplex MyComplex(double re, double im)
{
  dcomplex c;
  c.r = re;
  c.i = im;
  return c;
}

dcomplex MyCadd(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r + b.r;
  c.i = a.i + b.i;
  return c;
}


dcomplex MyCmul(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r*b.r - a.i*b.i;
  c.i = a.i*b.r + a.r*b.i;
  return c;
}



dcomplex MyCdiv(dcomplex a, dcomplex b)
{
  dcomplex c;
  double r,den;
  if (fabs(b.r) >= fabs(b.i)){
    r = b.i/b.r;
    den = b.r + r*b.i;
    c.r = (a.r + r*a.i)/den;
    c.i = (a.i - r*a.r)/den;
  }
  else{
    r = b.r/b.i;
    den = b.i + r*b.r;
    c.r = (a.r*r + a.i)/den;
    c.i = (a.i*r - a.r)/den;
  }
  return c;
}

dcomplex MyCsub(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r - b.r;
  c.i = a.i - b.i;
  return c;
}


dcomplex MyRCdiv(double x, dcomplex a)
{
  dcomplex c;
  double xx,yy,w;
  xx = a.r;
  yy = a.i;
  w = xx*xx+yy*yy;
  c.r = x*a.r/w;
  c.i = -x*a.i/w;
  return c;
}

