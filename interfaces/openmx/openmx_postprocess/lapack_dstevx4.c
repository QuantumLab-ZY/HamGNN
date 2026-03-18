/**********************************************************************
    lapack_dstevx4.c:

    lapack_dstevx4.c is a subroutine to find eigenvalues and eigenvectors
    of tridiagonlized real matrix using lapack's routines dstevx.

    Log of lapack_dstevx4.c:

       March/11/2013  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


void lapack_dstevx4(INTEGER N, INTEGER IL, INTEGER IU, double *D, double *E, double *W, double **ev)
{
  int i,j;
  char  *JOBZ="V";
  char  *RANGE="I";
  double VL,VU; /* dummy */
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;
  double *Z;
  INTEGER LDZ;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL;
  INTEGER INFO;
  
  M = IU - IL + 1;
  LDZ = N;

  Z = (double*)malloc(sizeof(double)*LDZ*M);
  WORK = (double*)malloc(sizeof(double)*5*N);
  IWORK = (INTEGER*)malloc(sizeof(INTEGER)*5*N);
  IFAIL = (INTEGER*)malloc(sizeof(INTEGER)*N);

  F77_NAME(dstevx,DSTEVX)( JOBZ, RANGE, &N, D, E, &VL, &VU, &IL, &IU, &ABSTOL,
           &M, W, Z, &LDZ, WORK, IWORK, IFAIL, &INFO );

  /* store eigenvectors */

  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      ev[i+IL][j+1]= Z[i*N+j];
    }
  }

  /* shift ko by 1 */
  for (i=M; i>=1; i--){
    W[i+IL-1]= W[i-1];
  }

  if (INFO>0) {
    /*
    printf("\n error in dstevx_, info=%d\n\n",INFO);fflush(stdout);
    */
  }
  if (INFO<0) {
    printf("info=%d in dstevx_\n",INFO);fflush(stdout);
    MPI_Finalize();
    exit(0);
  }

  free(Z);
  free(WORK);
  free(IWORK);
  free(IFAIL);
}
