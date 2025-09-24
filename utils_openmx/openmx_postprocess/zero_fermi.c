/******************************************************************

  zero_fermi.c generates zero points and associated residues
  of a continued fraction expansion terminated at M level
  which is derived from a hypergeometric function.

  This code is distributed under the constitution of GNU-GPL.

  (C) Taisuke Ozaki (AIST-RICS)

  The code and tables for the poles and residues can be downloadable
  from http://staff.aist.go.jp/t-ozaki/ 
  
  Log of zero_fermi.c:

     26/Nov/2006  Released by T.Ozaki


  **** HOW TO COMPILE *****

  For example, if there is an ATLAS library, libatlas_p4.a, 
  in a directory, /home/ozaki/lib, then compile. 

  % gcc zero_fermi.c -lm -L/home/ozaki/lib -latlas_p4 -o zero_fermi

  Binary files for ATLAS can be found in 
  http://www.theochem.ruhr-uni-bochum.de/~axel.kohlmeyer/cpmd-linux.html

  **** USAGE ****

  % ./zero_fermi 4

  where '4' means the number of poles of the continued fraction of 
  the Fermi-Dirac function on the upper half complex plane. 

  **** OUTPUT ****

   [ozaki@vtpcc01 exp]$ ./zero_fermi 4

              pole                  residue

    1  3.14159265364309e+00  -1.00000000028333e+00
    2  9.42675965413364e+00  -1.00295747791527e+00
    3  1.66063154702243e+01  -1.56204667295638e+00
    4  4.63195086818196e+01  -1.44349958488449e+01

  The 1st column: serial number
  The 2nd column: the imaginary part of pole, 
                  note that the real is zero. 
  The 3rd column: residue

******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <search.h>
#include <string.h>


int main(int argc, char *argv[]) 
{
  int i,j,N,M;
  double **A,**B,*zp,*Rp;

  /* check input parameters */

  if (argc!=2){
    printf("\ncould not find the number of zeros\n\n");
    exit(0);
  }

  /* find the number of zeros */

  N = (int)atof(argv[1]);
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

  zp = (double*)malloc(sizeof(double)*(M+2));
  Rp = (double*)malloc(sizeof(double)*(M+2));

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
  static long int itype=1;
  static long int n,lda,ldb,lwork,info;
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

  dsygv_( &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, zp, work, &lwork, &info);

  /*
  printf("info=%2d\n",info);
  */

  /* shift zp by 1 */
  for (i=n; i>=1; i--){
    zp[i]= zp[i-1];
  }

  /* store residue */

  for (i=1; i<=n; i++){
    zp[i] = 1.0/zp[i];
  }

  for (i=0; i<n; i++) {
    Rp[i+1] = -a[i*n]*a[i*n]*zp[i+1]*zp[i+1]*0.250;
  }

  free(a);
  free(b);
  free(work);
  }

  /* print result */

  printf("\n              pole                  residue\n\n");

  for (i=1; i<=N; i++){
    printf("%5d  %18.14e  %18.14e\n",i,-zp[i],Rp[i]);
  }

  /* free of arrays */

  for (i=0; i<(M+2); i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<(M+2); i++){
    free(B[i]);
  }
  free(B);

  free(zp);
  free(Rp);

  return 0;
}
