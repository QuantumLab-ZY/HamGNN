/**********************************************************************
    lapack_dstegr2.c:

    lapack_dstegr2.c is a subroutine to find eigenvalues and eigenvectors
    of tridiagonlized real matrix using lapack's routines dstegr.

    Log of lapack_dstegr2.c:

       Dec/24/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


void lapack_dstegr2(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, dcomplex **ev)
{
  int i,j;

  char  *JOBZ="V";
  char  *RANGE="A";
    
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL_GR=LAPACK_ABSTOL;
  INTEGER M;
  double *Z;
  INTEGER LDZ;
  INTEGER *ISUPPZ;
  INTEGER LWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER LIWORK;
  INTEGER INFO;

  IL = 1;
  IU = N;

  M = IU - IL + 1;
  LDZ = N;

  Z = (double*)malloc(sizeof(double)*LDZ*N);
  ISUPPZ = (INTEGER*)malloc(sizeof(INTEGER)*2*M);

  LWORK = 18*N;
  LIWORK = 10*N;
  WORK = (double*)malloc(sizeof(double)*LWORK);
  IWORK = (INTEGER*)malloc(sizeof(INTEGER)*LIWORK);

  F77_NAME(dstegr,DSTEGR)( JOBZ, RANGE, &N,  D, E, &VL, &VU, &IL, &IU, &ABSTOL_GR,
           &M, W, Z, &LDZ, ISUPPZ, WORK, &LWORK, IWORK, &LIWORK, &INFO );

  /* store eigenvectors */

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      ev[i+1][j+1].r = Z[i*N+j];
      ev[i+1][j+1].i = 0.0;
    }
  }

  /* shift ko by 1 */

  for (i=N; i>=1; i--){
    W[i]= W[i-1];
  }

  if (INFO>0) {
    /*
    printf("\n error in dstegr_, info=%d\n\n",INFO);
    */
  }
  if (INFO<0) {
    printf("info=%d in dstegr_\n",INFO);
    MPI_Finalize();
    exit(0);
  }


  free(Z);
  free(ISUPPZ);
  free(WORK);
  free(IWORK);

}
