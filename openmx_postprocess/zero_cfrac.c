/******************************************************************

  zero_cfrac.c generates zero points and associated residues
  of a continued fraction expansion terminated at 2n+2 level
  of the Ferm-Dirac function, which is derived from 
  a hypergeometric function.

  This code is distributed under the constitution of GNU-GPL.

 (C) Taisuke Ozaki (AIST-RICS)
  
  Log of zero_cfrac.c:

     14/July/2005  Released by T.Ozaki

******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <search.h>
#include <string.h>

#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


static void Eigen_DGGEVX( int n, double **a, double **s, double *eval,
                          double *resr, double *resi );

void zero_cfrac( int N, dcomplex *zp, dcomplex *Rp ) 
{

  int i,j,M;
  double **A,**B,*zp0,*Rp0;

  /* find the number of zeros */

  M = 2*N;

  /* allocation of arrays */

  A = (double**)malloc(sizeof(double*)*(M+2));
  for (i=0; i<(M+2); i++){
    A[i] = (double*)malloc(sizeof(double)*(M+2));
  }

  B = (double**)malloc(sizeof(double*)*(M+2));
  for (i=0; i<(M+2); i++){
    B[i] = (double*)malloc(sizeof(double)*(M+2));
  }

  zp0 = (double*)malloc(sizeof(double)*(M+2));
  Rp0 = (double*)malloc(sizeof(double)*(M+2));

  /* initialize arrays */

  for (i=0; i<(M+2); i++){
    for (j=0; j<(M+2); j++){
      A[i][j] = 0.0;
      B[i][j] = 0.0;
    }
  }

  /* set matrix elements */

  for (i=1; i<=M; i++){
    B[i][i] = (2.0*(double)i - 1.0);
  }

  for (i=1; i<=(M-1); i++){
    A[i][i+1] = -0.5;
    A[i+1][i] = -0.5;
  }

  /* diagonalization */

  {
    int i,j;
    char jobz = 'V';
    char uplo ='U';
    static INTEGER itype=1;
    static INTEGER n,lda,ldb,lwork,info;
    double *a,*b;
    double *work;

    n = M;
    lda = M;
    ldb = M;
    lwork = 3*M;

    a = (double*)malloc(sizeof(double)*n*n);
    b = (double*)malloc(sizeof(double)*n*n);
    work = (double*)malloc(sizeof(double)*3*n);

    for (i=0; i<n; i++) {
      for (j=0; j<n; j++) {
	a[j*n+i] = A[i+1][j+1];
	b[j*n+i] = B[i+1][j+1];
      }
    }

    F77_NAME(dsygv,DSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, zp0, work, &lwork, &info);

    /*
  printf("info=%2d\n",info);
    */

    /* shift zp0 by 1 */
    for (i=n; i>=1; i--){
      zp0[i]= zp0[i-1];
    }

    /* store residue */

    for (i=1; i<=n; i++){
      zp0[i] = -1.0/zp0[i];
    }

    for (i=0; i<n; i++) {
      Rp0[i+1] = -a[i*n]*a[i*n]*zp0[i+1]*zp0[i+1]*0.250;
    }

    free(a);
    free(b);
    free(work);
  }

  for (i=1; i<=N; i++){
    zp[i-1].r = 0.0;
    zp[i-1].i = zp0[i];
    Rp[i-1].r = Rp0[i];
    Rp[i-1].i = 0.0;
  }

  /* print result */

  /*
  for (i=0; i<N; i++){
    printf("i=%5d  zp=%18.14e Rp=%18.14e\n",i,zp[i].i,Rp[i].r);
  }

  MPI_Finalize();
  exit(0);
  */

  /* free of arrays */

  for (i=0; i<(M+2); i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<(M+2); i++){
    free(B[i]);
  }
  free(B);

  free(zp0);
  free(Rp0);

}




void zero_cfrac2( int n, dcomplex *zp, dcomplex *Rp ) 
{
  static int i,j,N;
  static double **a,**s,*eval,*resr,*resi;

  /* check input parameters */

  if (n<=0){
    printf("\ncould not find the number of zeros\n\n");
    MPI_Finalize();
    exit(0);
  }

  /* the total number of zeros including minus value */

  N = 2*n + 1;

  /* allocation of arrays */

  a = (double**)malloc(sizeof(double*)*(N+2));
  for (i=0; i<(N+2); i++){
    a[i] = (double*)malloc(sizeof(double)*(N+2));
  }

  s = (double**)malloc(sizeof(double*)*(N+2));
  for (i=0; i<(N+2); i++){
    s[i] = (double*)malloc(sizeof(double)*(N+2));
  }

  eval = (double*)malloc(sizeof(double)*(n+3));
  resr = (double*)malloc(sizeof(double)*(n+3));
  resi = (double*)malloc(sizeof(double)*(n+3));

  /* initialize arrays */

  for (i=0; i<(N+2); i++){
    for (j=0; j<(N+2); j++){
      a[i][j] = 0.0;
      s[i][j] = 0.0;
    }
  }

  /* set matrix elements */

  s[2][1] =  1.0;
  s[2][2] = -0.5;

  for (i=3; i<=N; i++){
     s[i][i-1] =  -0.5;
     s[i-1][i] =   0.5;
  }

  a[1][1] = -2.0;
  a[1][2] =  1.0;
  a[2][2] = -1.0;

  for (i=3; i<=N; i++){
    a[i][i] = -(2.0*(double)i - 3.0);
  }

  /* diagonalization */

  Eigen_DGGEVX( N, a, s, eval, resr, resi );

  for (i=0; i<n; i++){
    zp[i].r = 0.0;
    zp[i].i = eval[i+1];
    Rp[i].r = resr[i+1];
    Rp[i].i = 0.0;
  }

  /* print result */

  for (i=0; i<n; i++){
    printf("i=%5d  zp=%18.14e Rp=%18.14e\n",i,zp[i].i,Rp[i].r);
  }

  MPI_Finalize();
  exit(0);
  

  /* free of arrays */

  for (i=0; i<(N+2); i++){
    free(a[i]);
  }
  free(a);

  for (i=0; i<(N+2); i++){
    free(s[i]);
  }
  free(s);

  free(eval);
  free(resr);
  free(resi);
}













void Eigen_DGGEVX( int n, double **a, double **s, double *eval, double *resr, double *resi )
{
  static int i,j,k,l,num;

  static char balanc = 'N';
  static char jobvl = 'V';
  static char jobvr = 'V';
  static char sense = 'B';
  static double *A;
  static double *b;
  static double *alphar;
  static double *alphai;
  static double *beta;
  static double *vl;
  static double *vr;
  static double *lscale;
  static double *rscale;
  static double abnrm;
  static double bbnrm;
  static double *rconde;
  static double *rcondv;
  static double *work;
  static double *tmpvecr,*tmpveci;
  static INTEGER *iwork;
  static INTEGER lda,ldb,ldvl,ldvr,ilo,ihi;
  static INTEGER lwork,info;
  static logical *bwork; 
  static double sumr,sumi,tmpr,tmpi;
  static double *sortnum;

  lda = n;
  ldb = n;
  ldvl = n;
  ldvr = n;

  A = (double*)malloc(sizeof(double)*n*n);
  b = (double*)malloc(sizeof(double)*n*n);
  alphar = (double*)malloc(sizeof(double)*n);
  alphai = (double*)malloc(sizeof(double)*n);
  beta = (double*)malloc(sizeof(double)*n);

  vl = (double*)malloc(sizeof(double)*n*n);
  vr = (double*)malloc(sizeof(double)*n*n);

  lscale = (double*)malloc(sizeof(double)*n);
  rscale = (double*)malloc(sizeof(double)*n);

  rconde = (double*)malloc(sizeof(double)*n);
  rcondv = (double*)malloc(sizeof(double)*n);

  lwork = 2*n*n + 12*n + 16;
  work = (double*)malloc(sizeof(double)*lwork);

  iwork = (INTEGER*)malloc(sizeof(INTEGER)*(n+6));
  bwork = (logical*)malloc(sizeof(logical)*n);

  tmpvecr = (double*)malloc(sizeof(double)*(n+2));
  tmpveci = (double*)malloc(sizeof(double)*(n+2));

  sortnum = (double*)malloc(sizeof(double)*(n+2));

  /* convert two dimensional arrays to one-dimensional arrays */

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
       A[j*n+i]= a[i+1][j+1];
       b[j*n+i]= s[i+1][j+1];
    }
  }

  /* call dggevx_() */

  F77_NAME(dggevx,DGGEVX)(
           &balanc, &jobvl, & jobvr, &sense, &n, A, &lda, b, &ldb,
           alphar, alphai, beta, vl, &ldvl, vr, &ldvr, &ilo, &ihi,
           lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work,
           &lwork, iwork, bwork, &info );

  if (info!=0){
    printf("Errors in dggevx_() info=%2d\n",info);
  }

  /*
  for (i=0; i<n; i++){
    printf("i=%4d  %18.13f %18.13f %18.13f\n",i,alphar[i],alphai[i],beta[i]);
  }
  printf("\n");
  */

  num = 0;
  for (i=0; i<n; i++){

    if ( 1.0e-13<fabs(beta[i]) && 0.0<alphai[i]/beta[i] ){

      /* normalize the eigenvector */

      for (j=0; j<n; j++) {

        sumr = 0.0;
        sumi = 0.0;

        for (k=0; k<n; k++) {
          sumr += s[j+1][k+1]*vr[i*n    +k];
          sumi += s[j+1][k+1]*vr[(i+1)*n+k];
        }
        
        tmpvecr[j] = sumr;
        tmpveci[j] = sumi;
      }

      sumr = 0.0;
      sumi = 0.0;

      for (k=0; k<n; k++) {
        sumr += vl[i*n+k]*tmpvecr[k] + vl[(i+1)*n+k]*tmpveci[k];
        sumi += vl[i*n+k]*tmpveci[k] - vl[(i+1)*n+k]*tmpvecr[k];
      }

      /* calculate zero point and residue */

      eval[num+1] = alphai[i]/beta[i];
      tmpr =  vr[i*n]*vl[i*n] + vr[(i+1)*n]*vl[(i+1)*n];
      tmpi = -vr[i*n]*vl[(i+1)*n] + vr[(i+1)*n]*vl[i*n];
      resr[num+1] =  tmpi/sumi;
      resi[num+1] = -tmpr/sumi;

      num++;
    }
    else{
      /*
      printf("i=%4d  %18.13f %18.13f %18.13f\n",i+1,alphar[i],alphai[i],beta[i]);
      */
    }
  }

  /* check round-off error */

  for (i=1; i<=num; i++){
    if (1.0e-8<fabs(resi[i])){
      printf("Could not calculate zero points and residues due to round-off error\n");
      MPI_Finalize();
      exit(0);
    }
  }

  /* sorting */

  qsort_double(num,eval,resr);

  /* free arraies */

  free(A);
  free(b);
  free(alphar);
  free(alphai);
  free(beta);

  free(vl);
  free(vr);

  free(lscale);
  free(rscale);

  free(rconde);
  free(rcondv);

  free(work);

  free(iwork);
  free(bwork);

  free(tmpvecr);
  free(tmpveci);
  free(sortnum);
}

