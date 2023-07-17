/**********************************************************************
  lapack_dstevx2.c:

  lapack_dstevx2.c is a subroutine to find eigenvalues and eigenvectors
  of tridiagonlized real matrix using lapack's routines dstevx.

  Log of lapack_dstevx2.c:

     Dec/24/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


void lapack_dstevx2(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, dcomplex **ev, int ev_flag)
{
  int i,j;

  char  *JOBZN="N";
  char  *JOBZV="V";
  char  *RANGE="I";

  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;
  double *Z;
  INTEGER LDZ;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL;
  INTEGER INFO;

  IL = 1;
  IU = EVmax;

  M = IU - IL + 1;
  LDZ = N;

  Z = (double*)malloc(sizeof(double)*LDZ*N);
  WORK = (double*)malloc(sizeof(double)*5*N);
  IWORK = (INTEGER*)malloc(sizeof(INTEGER)*5*N);
  IFAIL = (INTEGER*)malloc(sizeof(INTEGER)*N);

  if (ev_flag){
    F77_NAME(dstevx,DSTEVX)( JOBZV, RANGE, &N,  D, E, &VL, &VU, &IL, &IU, &ABSTOL,
             &M, W, Z, &LDZ, WORK, IWORK, IFAIL, &INFO );
  }
  else{
    F77_NAME(dstevx,DSTEVX)( JOBZN, RANGE, &N,  D, E, &VL, &VU, &IL, &IU, &ABSTOL,
             &M, W, Z, &LDZ, WORK, IWORK, IFAIL, &INFO );
  }

  /* store eigenvectors */

  if (ev_flag==1){
    for (i=0; i<EVmax; i++) {
      for (j=0; j<N; j++) {
        ev[i+1][j+1].r = Z[i*N+j];
        ev[i+1][j+1].i = 0.0;
      }
    }
  }

  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    W[i]= W[i-1];
  }

  if (INFO>0) {
    /*
    printf("\n error in dstevx_, info=%d\n\n",INFO);
    */
  }
  if (INFO<0) {
    printf("info=%d in dstevx_\n",INFO);
    MPI_Finalize();
    exit(0);
  }

  free(Z);
  free(WORK);
  free(IWORK);
  free(IFAIL);
}
