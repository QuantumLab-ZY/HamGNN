/*
Useful functions and subroutines optimized for jx.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <string.h>
//#include <malloc/malloc.h>
//#include <assert.h>

#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"

#include "jx_tools.h"

void matmul_dcomplex_lapack(
  char *typeA, char *typeB, int m, int n, int k,
  dcomplex **A_2d, dcomplex **B_2d, dcomplex **C_2d )
{

  static dcomplex *A_1d,*B_1d,*C_1d;
  static dcomplex dcplx1;
  static dcomplex dcplx0;
  static int i,j;
  static int nrow_A,nrow_B,ncol_A,ncol_B;

//  printf("%i %i %i\n",m,n,k);

  dcplx1.r = 1.0;
  dcplx1.i = 0.0;
  dcplx0.r = 0.0;
  dcplx0.i = 0.0;

  A_1d = (dcomplex*)malloc(sizeof(dcomplex)*m*k);
  B_1d = (dcomplex*)malloc(sizeof(dcomplex)*k*n);
  C_1d = (dcomplex*)malloc(sizeof(dcomplex)*m*n);

  if (strcmp(typeA,"N")==0){
    nrow_A=m;
    ncol_A=k;
  }
  else{
    nrow_A=k;
    ncol_A=m;
  }

  for(i=0; i<nrow_A; i++){
    for(j=0; j<ncol_A; j++){
//      printf("%i %i %f \n",i,j,A_2d[i+1][j+1].r);
      A_1d[i+nrow_A*j]=A_2d[i+1][j+1];
    }
  }

//  printf("68\n");

  if (strcmp(typeB,"N")==0){
    nrow_B=k;
    ncol_B=n;
  }
  else{
    nrow_B=n;
    ncol_B=k;
  }

  for(i=0; i<nrow_B; i++){
    for(j=0; j<ncol_B; j++){
//      printf("%i %i %i \n",i+1,j+1,i+nrow_B*j);
      B_1d[i+nrow_B*j]=B_2d[i+1][j+1];
    }
  }

//  printf("85\n");
//  printf("%i %i %i %i\n",n,k,nrow_B,ncol_B);

  F77_NAME(zgemm,ZGEMM)(
    typeA,typeB,&m,&n,&k,
    &dcplx1,
    A_1d,&nrow_A,
    B_1d,&nrow_B,
    &dcplx0,
    C_1d,&m);

  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      C_2d[i+1][j+1]=C_1d[i+m*j];
    }
  }

  free(A_1d);
  free(B_1d);
  free(C_1d);

}

void matmul_double_lapack(
  char *typeA, char *typeB, int m, int n, int k,
  double **A_2d, double **B_2d, double **C_2d )
{

  static double *A_1d,*B_1d,*C_1d;
  static double dble1;
  static double dble0;
  static int i,j;
  static int nrow_A,nrow_B,ncol_A,ncol_B;

  dble1 = 1.0;
  dble0 = 0.0;

  A_1d = (double*)malloc(sizeof(double)*m*k);
  B_1d = (double*)malloc(sizeof(double)*k*n);
  C_1d = (double*)malloc(sizeof(double)*m*n);

  if (strcmp(typeA,"N")==0){
    nrow_A=m;
    ncol_A=k;
  }
  else{
    nrow_A=k;
    ncol_A=m;
  }

  for(i=0; i<nrow_A; i++){
    for(j=0; j<ncol_A; j++){
      A_1d[i+nrow_A*j]=A_2d[i+1][j+1];
    }
  }

  if (strcmp(typeB,"N")==0){
    nrow_B=k;
    ncol_B=n;
  }
  else{
    nrow_B=n;
    ncol_B=k;
  }

  for(i=0; i<nrow_B; i++){
    for(j=0; j<ncol_B; j++){
      B_1d[i+nrow_B*j]=B_2d[i+1][j+1];
    }
  }

  F77_NAME(dgemm,DGEMM)(
    typeA,typeB,
    &m,&n,&k,&dble1,
    A_1d,&nrow_A,
    B_1d,&nrow_B,
    &dble0,
    C_1d,&m);

  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      C_2d[i+1][j+1]=C_1d[i+m*j];
    }
  }

  free(A_1d);
  free(B_1d);
  free(C_1d);

}

//
// この下オリジナルからコピペ
//

void k_inversion(int i,  int j,  int k,
                 int mi, int mj, int mk,
                 int *ii, int *ij, int *ik )
{
  *ii= mi-i-1;
  *ij= mj-j-1;
  *ik= mk-k-1;
}

void Overlap_Band(double ****OLP,
                  dcomplex **S,int *MP,
                  double k1, double k2, double k3)
{
  static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  static int l1,l2,l3,Rn,n2;
  static double **S1,**S2;
  static double kRn,si,co,s;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  S1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S1[i] = (double*)malloc(sizeof(double)*n2);
  }

  S2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                       set overlap
  ****************************************************/

  S[0][0].r = NUM;

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S1[i][j] = 0.0;
      S2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  s = OLP[GA_AN][LB_AN][i][j];
	  S1[Anum+i][Bnum+j] = S1[Anum+i][Bnum+j] + s*co;
	  S2[Anum+i][Bnum+j] = S2[Anum+i][Bnum+j] + s*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S[i][j].r =  S1[i][j];
      S[i][j].i =  S2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(S1[i]);
    free(S2[i]);
  }
  free(S1);
  free(S2);

}


void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
                      double k1, double k2, double k3)
{
  static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  static int l1,l2,l3,Rn,n2;
  static double **H1,**H2;
  static double kRn,si,co,h;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  H1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H1[i] = (double*)malloc(sizeof(double)*n2);
  }

  H2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                    set Hamiltonian
  ****************************************************/

  H[0][0].r = 2.0*NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H1[i][j] = 0.0;
      H2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  h = RH[GA_AN][LB_AN][i][j];
	  H1[Anum+i][Bnum+j] = H1[Anum+i][Bnum+j] + h*co;
	  H2[Anum+i][Bnum+j] = H2[Anum+i][Bnum+j] + h*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i][j].r = H1[i][j];
      H[i][j].i = H2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H1[i]);
    free(H2[i]);
  }
  free(H1);
  free(H2);
}

void Eigen_lapack(double **a, double *ko, int n)
{
  /* input:  n;
     input:  a[n][n];  matrix A

     output: a[n][n]; eigevectors
     output: ko[n];   eigenvalues  */

  static char *name="Eigen_lapack";

  char  *JOBZ="V";
  char  *RANGE="A";
  char  *UPLO="L";

  int LDA=n;
  double VL,VU; /* dummy */
  int IL,IU; /* dummy */
  double ABSTOL=1.0e-10;
  int M;

  double *A,*Z;
  int LDZ=n;
  int LWORK;
  double *WORK;
  int *IWORK;

  int *IFAIL, INFO;

  int i,j;

  A=(double*)malloc(sizeof(double)*n*n);
  Z=(double*)malloc(sizeof(double)*n*n);

  LWORK=n*8;
  WORK=(double*)malloc(sizeof(double)*LWORK);
  IWORK=(int*)malloc(sizeof(int)*n*5);
  IFAIL=(int*)malloc(sizeof(int)*n);

  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      A[i*n+j]= a[i+1][j+1];
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

  /* store eigenvectors */
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      /*  a[i+1][j+1]= Z[i*n+j]; */
      a[j+1][i+1]= Z[i*n+j];

    }
  }

  /* shift ko by 1 */
  for (i=n;i>=1;i--){
    ko[i]= ko[i-1];
  }

  if (INFO>0) {
    printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO);
  }
  if (INFO<0) {
    printf("%s: info=%d\n",name,INFO);
    exit(10);
  }

  free(IFAIL); free(IWORK); free(WORK); free(Z); free(A);

}

void EigenBand_lapack(dcomplex **A, double *W, int N)
{
  static char *JOBZ="V";
  static char *UPLO="L";
  int LWORK;
  dcomplex *A0;
  dcomplex *WORK;
  double *RWORK;
  int INFO;
  int i,j;

  A0=(dcomplex*)malloc(sizeof(dcomplex)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A0[(j-1)*N+i-1] = A[i][j];
    }
  }

  LWORK=3*N;
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK=(double*)malloc(sizeof(double)*(3*N-2));
  F77_NAME(zheev,ZHEEV)(JOBZ,UPLO, &N, A0, &N, W, WORK, &LWORK, RWORK, &INFO  );

  if (INFO!=0) {
    printf("************************************************************\n");
    printf("  EigenBand_lapack: cheev_()=%d\n",INFO);
    printf("************************************************************\n");
    exit(10);
  }

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A[i][j].r = A0[(j-1)*N+i-1].r;
      A[i][j].i = A0[(j-1)*N+i-1].i;
    }
  }

  for (i=N;i>=1;i--) {
    W[i] =W[i-1];
  }

  free(A0); free(RWORK); free(WORK);
}

void matinv_double_lapack(double **A, int N)
{

  double *A_1d;
  int LWORK;
  double *WORK;
//  double *RWORK;
  int *IPIV;
  int INFO;

  int i,j;

  A_1d=(double*)malloc(sizeof(double)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A_1d[(j-1)*N+i-1] = A[i][j];
    }
  }

  LWORK=3*N;
  WORK=(double*)malloc(sizeof(double)*LWORK);
  IPIV=(int*)malloc(sizeof(int)*N);
  F77_NAME(dgetrf,DGETRF)( &N, &N, A_1d, &N, IPIV, &INFO);
  F77_NAME(dgetri,DGETRI)( &N, A_1d, &N, IPIV, WORK, &LWORK, &INFO);
  free(IPIV);
//  free(RWORK);
  free(WORK);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A[i][j] = A_1d[(j-1)*N+i-1];
    }
  }

  free(A_1d);
}

void matinv_dcplx_lapack(dcomplex **A, int N)
{

  dcomplex *A_1d;
  int LWORK;
  dcomplex *WORK;
//  double *RWORK;
  int *IPIV;
  int INFO;

  int i,j;

  A_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A_1d[(j-1)*N+i-1] = A[i][j];
    }
  }

  LWORK=3*N;
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  IPIV=(int*)malloc(sizeof(int)*N);
  F77_NAME(zgetrf,ZGETRF)( &N, &N, A_1d, &N, IPIV, &INFO);
  F77_NAME(zgetri,ZGETRI)( &N, A_1d, &N, IPIV, WORK, &LWORK, &INFO);
  free(IPIV);
//  free(RWORK);
  free(WORK);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A[i][j].r = A_1d[(j-1)*N+i-1].r;
      A[i][j].i = A_1d[(j-1)*N+i-1].i;
    }
  }

  free(A_1d);
}

void dtime(double *t)
{
  /* real time */
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

  /* user time + system time */
  /*
    float tarray[2];
    clock_t times(), wall;
    struct tms tbuf;
    wall = times(&tbuf);
    tarray[0] = (float) (tbuf.tms_utime / (float)CLOCKS_PER_SEC);
    tarray[1] = (float) (tbuf.tms_stime / (float)CLOCKS_PER_SEC);
    *t = (double) (tarray[0]+tarray[1]);
    printf("dtime: %lf\n",*t);
  */
}

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

void gen_eigval_herm(dcomplex **A, dcomplex **B, double *V, int N)
{

  dcomplex *A_1d, *B_1d, *C_1d, *D_1d;
  double *A_eig, *B_eig;

  int LWORK;
  dcomplex *WORK;
  double *RWORK;
  static dcomplex dcplx1;
  static dcomplex dcplx0;

  int INFO;

  int i,j,k,l;
  double temp;

  dcplx1.r = 1.0;
  dcplx1.i = 0.0;
  dcplx0.r = 0.0;
  dcplx0.i = 0.0;

  A_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);
  B_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);
  C_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);
  D_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A_1d[(j-1)*N+i-1] = A[i][j];
      B_1d[(j-1)*N+i-1] = B[i][j];
    }
  }

  A_eig=(double*)malloc(sizeof(double)*N);
  B_eig=(double*)malloc(sizeof(double)*N);

  LWORK=3*N;
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK=(double*)malloc(sizeof(double)*(3*N-2));

/*  printf("\n");
  printf("A = \n");
  for (i=1; i<=N; i++){
    for (j=1; j<=N; j++){
      printf("%f ",A[i][j].r);
    }
    printf("\n");
  }

  printf("\n");
  printf("B = \n");
  for (i=1; i<=N; i++){
    for (j=1; j<=N; j++){
      printf("%f ",B[i][j].r);
    }
    printf("\n");
  } */

  F77_NAME(zheev,ZHEEV)(
    "V","L", &N, B_1d, &N, B_eig, WORK, &LWORK, RWORK, &INFO  );

/*    printf("\n");
    printf("L_B = \n");
    for (i=1; i<=N; i++){
      printf("%f ",B_eig[i-1]);
    }
    printf("\n");

  printf("\n");
  printf("U_B = \n");
  for (i=1; i<=N; i++){
    for (j=1; j<=N; j++){
      printf("%f ",B_1d[(j-1)*N+i-1].r);
    }
    printf("\n");
  }

    printf("\n");
    printf("U_B^TBU_B = \n");
    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        temp=0.0;
        for (k=1; k<=N; k++){
          for (l=1; l<=N; l++){
            temp+=B_1d[(i-1)*N+k-1].r*B[k][l].r*B_1d[(j-1)*N+l-1].r;
          }
        }
        printf("%f ",temp);
      }
      printf("\n");
    } */


  F77_NAME(zgemm,ZGEMM)(
    "C","N",&N,&N,&N,&dcplx1,
    B_1d,&N,A_1d,&N,&dcplx0,C_1d,&N);

/*    printf("\n");
    printf("U_B^TA = \n");

    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        printf("%f ",C_1d[(j-1)*N+i-1].r);
      }
      printf("\n");
    }

    printf("\n");
    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        temp=0.0;
        for (k=1; k<=N; k++){
            temp+=B_1d[(i-1)*N+k-1].r*A_1d[(j-1)*N+k-1].r;
        }
        printf("%f ",temp);
      }
      printf("\n");
    } */

  F77_NAME(zgemm,ZGEMM)(
    "N","N",&N,&N,&N,&dcplx1,
    C_1d,&N,B_1d,&N,&dcplx0,D_1d,&N);

/*    printf("\n");
    printf("U_B^TAU_B = \n");

    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        printf("%f ",D_1d[(j-1)*N+i-1].r);
      }
      printf("\n");
    }

    printf("\n");
    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        temp=0.0;
        for (k=1; k<=N; k++){
            temp+=C_1d[(k-1)*N+i-1].r*B_1d[(j-1)*N+k-1].r;
        }
        printf("%f ",temp);
      }
      printf("\n");
    } */

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A_1d[(j-1)*N+i-1].r
      = D_1d[(j-1)*N+i-1].r/(sqrt(B_eig[i-1])*sqrt(B_eig[j-1]));
      A_1d[(j-1)*N+i-1].i
      = D_1d[(j-1)*N+i-1].i/(sqrt(B_eig[i-1])*sqrt(B_eig[j-1]));
    }
  }

/*  printf("\n");
  printf("A'=invsqrt(L_B)U_B^TAU_Binvsqrt(L_B) = \n");
  for (i=1; i<=N; i++){
    for (j=1; j<=N; j++){
      printf("%f ",A_1d[(j-1)*N+i-1].r);
    }
    printf("\n");
  } */

  free(WORK);
  free(RWORK);
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK=(double*)malloc(sizeof(double)*(3*N-2));

  F77_NAME(zheev,ZHEEV)(
    "V", "U", &N, A_1d, &N, A_eig, WORK, &LWORK, RWORK, &INFO  );

  free(WORK);
  free(RWORK);

/*    printf("\n");
    printf("L_A' = \n");
    for (i=1; i<=N; i++){
      printf("%f ",A_eig[i-1]);
    }
    printf("\n");

    printf("\n");
    printf("U_A' = \n");
    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        printf("%f ",A_1d[(j-1)*N+i-1].r);
      }
      printf("\n");
    }

    printf("\n");
    printf("U_A'^TU_A' = \n");
    for (i=1; i<=N; i++){
      for (j=1; j<=N; j++){
        temp=0.0;
        for (k=1; k<=N; k++){
            temp+=A_1d[(i-1)*N+k-1].r*A_1d[(j-1)*N+k-1].r;
        }
        printf("%f ",temp);
      }
      printf("\n");
    } */

    for (i=1;i<=N;i++) {
      for (j=1;j<=N;j++) {
        C_1d[(j-1)*N+i-1].r
        = A_1d[(j-1)*N+i-1].r/(sqrt(B_eig[i-1]));
        C_1d[(j-1)*N+i-1].i
        = A_1d[(j-1)*N+i-1].i/(sqrt(B_eig[i-1]));
      }
    }

  F77_NAME(zgemm,ZGEMM)(
    "N","N",&N,&N,&N,&dcplx1,
    B_1d,&N,C_1d,&N,&dcplx0,A_1d,&N);

  for (i=1;i<=N;i++) {
    V[i]=A_eig[i-1];
    for (j=1;j<=N;j++) {
      A[i][j].r = A_1d[(j-1)*N+i-1].r;
      A[i][j].i = A_1d[(j-1)*N+i-1].i;
    }
//    printf("\n");
  }

  free(A_1d);
  free(B_1d);
  free(C_1d);
  free(D_1d);
  free(A_eig);
  free(B_eig);

}

void gen_eigval_herm_lapack(dcomplex **A, dcomplex **B, double *V, int N)
{
  dcomplex *A_1d, *B_1d;
  double *A_eig;

  int ITYPE;
  int LWORK;
  dcomplex *WORK;
  double *RWORK;
  static dcomplex dcplx1;
  static dcomplex dcplx0;

  int INFO;

  int i,j;

  dcplx1.r = 1.0;
  dcplx1.i = 0.0;
  dcplx0.r = 0.0;
  dcplx0.i = 0.0;

  A_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);
  B_1d=(dcomplex*)malloc(sizeof(dcomplex)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A_1d[(j-1)*N+i-1] = A[i][j];
      B_1d[(j-1)*N+i-1] = B[i][j];
    }
  }

  A_eig=(double*)malloc(sizeof(double)*N);

  ITYPE=1;
  LWORK=3*N;
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK=(double*)malloc(sizeof(double)*(3*N-2));


  F77_NAME(zhegv,ZHEGV)(
    &ITYPE,"V","L", &N, A_1d, &N, B_1d, &N, A_eig, WORK, &LWORK, RWORK, &INFO  );


    for (i=1;i<=N;i++) {
      V[i]=A_eig[i-1];
      for (j=1;j<=N;j++) {
        A[i][j].r = A_1d[(j-1)*N+i-1].r;
        A[i][j].i = A_1d[(j-1)*N+i-1].i;
      }
    }

    free(WORK);
    free(RWORK);
    free(A_1d);
    free(B_1d);
    free(A_eig);

}

void dcplx_basis_transformation(
  int num, dcomplex **trns_mat_1, dcomplex **mat, dcomplex **trns_mat_2){

  int i,j,k;
  dcomplex **mat_1,**mat_2;

/*  printf("mat\n");
  for (i=0; i<=num; i++){
    for (j=0; j<=num; j++){
      printf("%i %i %f %f \n",i,j,mat[i][j].r, mat[i][j].i);
    }
  }
  printf("trns_mat_1\n");
  for (i=1; i<=num; i++){
    for (j=1; j<=num; j++){
      printf("%i %i %f %f \n",i,j,trns_mat_1[i][j].r, trns_mat_1[i][j].i);
    }
  }
  printf("trns_mat_2\n");
  for (i=1; i<=num; i++){
    for (j=1; j<=num; j++){
      printf("%i %i %f %f \n",i,j,trns_mat_2[i][j].r, trns_mat_2[i][j].i);
    }
  } */

  mat_1 = (dcomplex**)malloc( sizeof(dcomplex*)*(num+1) ) ;
  mat_2 = (dcomplex**)malloc( sizeof(dcomplex*)*(num+1) ) ;
  for (i=0; i<=num; i++){
    mat_1[i] = (dcomplex*)malloc( sizeof(dcomplex)*(num+1) );
    mat_2[i] = (dcomplex*)malloc( sizeof(dcomplex)*(num+1) );
  }
  for (i=0; i<=num; i++){
    for (j=0; j<=num; j++){
      mat_1[i][j].r = 0.0;
      mat_1[i][j].i = 0.0;
      mat_2[i][j].r = 0.0;
      mat_2[i][j].i = 0.0;
    }
  }

  matmul_dcomplex_lapack("N","N",num,num,num,trns_mat_1,mat,mat_1);
  matmul_dcomplex_lapack("N","N",num,num,num,mat_1,trns_mat_2,mat_2);

  for (i=1; i<=num; i++){
    for (j=1; j<=num; j++){
      mat[i][j] = mat_2[i][j];
    }
  }

  for (i=0; i<=num; i++){
    free(mat_1[i]);
    free(mat_2[i]);
  }
  free(mat_1);
  free(mat_2);

}
