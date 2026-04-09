/**********************************************************************
    Eigen_lapack3.c:

    Eigen_lapack3.c is a subroutine to solve a seqular equation without an
    overlap matrix using lapack's routines.

    Log of Eigen_lapack3.c:

       8/May/2014  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define  measure_time   0

static void Eigen_HH3(double *ac, int csize, double *ko, int n, int EVmax);
static int Eigen_lapack_x3(double *a, double *ko, int n0, int EVmax);

void Eigen_lapack3(double *a, double *ko, int n, int EVmax)
{
  int info;

  info = Eigen_lapack_x3(a, ko, n, EVmax);
} 






int Eigen_lapack_x3(double *a, double *ko, int n0, int EVmax)
{

  /*
    F77_NAME(dsyevx,DSYEVX)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  char *name="Eigen_lapack";

  char  *JOBZ="V";
  char  *RANGE="I";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n0;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;

  double *A,*Z;
  INTEGER LDZ=n;
  INTEGER LWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL, INFO;

  int i,j;

  A=(double*)malloc(sizeof(double)*n*n);
  Z=(double*)malloc(sizeof(double)*n*n);

  LWORK=n*8;
  WORK=(double*)malloc(sizeof(double)*LWORK);
  IWORK=(INTEGER*)malloc(sizeof(INTEGER)*n*5);
  IFAIL=(INTEGER*)malloc(sizeof(INTEGER)*n);

  IL = 1;
  IU = EVmax;
 
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      A[i*n+j] = a[i*n+j];
    }
  }

#if 0
  printf("A=\n");
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      printf("%f ",A[i*n+j]);
    }
    printf("\n");
  }
  fflush(stdout);
#endif

  F77_NAME(dsyevx,DSYEVX)( JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
			   &ABSTOL, &M, ko, Z, &LDZ, WORK, &LWORK, IWORK,
			   IFAIL, &INFO ); 

  if (INFO>0) {
    /* printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf("%s: info=%d\n",name,INFO);
    exit(10);
  }
  else{ /* (INFO==0) */
    /* store eigenvectors */
    for (i=0;i<EVmax;i++) {
      for (j=0;j<n;j++) {
	a[i*n+j]= Z[i*n+j];
      }
    }
  }
   
  free(IFAIL); free(IWORK); free(WORK); free(Z); free(A);

  return INFO;
}



