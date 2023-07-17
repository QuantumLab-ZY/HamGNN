/**********************************************************************
    lapack_dstegr1.c:

    lapack_dstegr1.c is a subroutine to find eigenvalues and eigenvectors
    of tridiagonlized real matrix using lapack's routine, dstegr.

    Log of lapack_dstevx1.c:

       Dec/24/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


void lapack_dsteqr1(INTEGER N, double *D, double *E, double *W, double **ev)
{
  int i,j;
  char  *COMPZ="I";
  double *Z;
  INTEGER LDZ;
  double *WORK;
  INTEGER INFO;

  LDZ = N;
  Z = (double*)malloc(sizeof(double)*LDZ*N);
  WORK = (double*)malloc(sizeof(double)*2*N);

  F77_NAME(dsteqr,DSTEQR)( COMPZ, &N, D, E, Z, &LDZ, WORK, &INFO );

  /* store eigenvectors */

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      ev[i+1][j+1]= Z[i*N+j];
    }
  }

  /* shift ko by 1 */
  for (i=N; i>=1; i--){
    W[i]= D[i-1];
  }
  
  if (INFO>0) {
    printf("\n error in dstevx_, info=%d\n\n",INFO);fflush(stdout);
  }
  if (INFO<0) {
    printf("info=%d in dstevx_\n",INFO);fflush(stdout);
    MPI_Finalize();
    exit(0);
  }

  free(Z);
  free(WORK);
}
