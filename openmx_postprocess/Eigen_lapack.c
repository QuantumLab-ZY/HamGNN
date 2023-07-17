/**********************************************************************
    Eigen_lapack.c:

    Eigen_lapack.c is a subroutine to solve a seqular equation without an
    overlap matrix using lapack's routines.

    Log of Eigen_lapack.c:

       Dec/10/2002  Released by H.Kino
       Nov/22/2004  Modified by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define  measure_time   0

static void Eigen_lapack_x(double **a, double *ko, int n, int EVmax);
void Eigen_lapack_d(double **a, double *ko, int n, int EVmax);
static void Eigen_lapack_r(double **a, double *ko, int n, int EVmax);
static void Eigen_HHQR(double **ac, double *ko, int n, int EVmax);
static void Eigen_HH(double **ac, double *ko, int n, int EVmax);
   
void Eigen_lapack(double **a, double *ko, int n, int EVmax)
{

  static int solver_flag=0;
  int i,j,k,po,okay_flag,iterN;
  double sum;
  double **b;

  b=(double**)malloc(sizeof(double*)*(n+1));
  for (i=0; i<(n+1); i++){
    b[i]=(double*)malloc(sizeof(double)*(n+1));
  }

  for (i=0; i<(n+1); i++){
    for (j=0; j<(n+1); j++){
      b[i][j] = a[i][j]; 
    }
  }

  iterN = 1;
  okay_flag = 0;

  do {

    for (i=0; i<(n+1); i++){
      for (j=0; j<(n+1); j++){
	a[i][j] = b[i][j]; 
      }
    }

    if      (solver_flag==0) Eigen_lapack_x(a, ko, n, EVmax);
    else if (solver_flag==1) Eigen_HH(a, ko, n, EVmax);  
    else if (solver_flag==2) Eigen_lapack_d(a, ko, n, EVmax); 

    po = 0; 

    i = 1;
    do {
      j = 1;
      do {

	sum = 0.0;
	for (k=1; k<=n; k++){
	  sum += a[i][k]*a[j][k]; 
	}

	if      (i==j && 0.00001<fabs(sum-1.0)){ po = 1; }
	else if (i!=j && 0.00001<fabs(sum))    { po = 1; }

        j = j + 20;

      } while (po==0 && j<=EVmax); 

      i = i + 20;

    } while (po==0 && i<=EVmax); 

    if (po==1){
      solver_flag++; 
      solver_flag = solver_flag % 3;
    }
    else {
      okay_flag = 1;
    }

    /*
    printf("iterN=%2d solver_flag=%2d po=%2d okay_flag=%2d\n",iterN,solver_flag,po,okay_flag);
    */

    iterN++;

  } while (okay_flag==0 && iterN<4);  

  for (i=0; i<(n+1); i++){
    free(b[i]);
  }
  free(b);

} 


void Eigen_lapack_x(double **a, double *ko, int n0, int EVmax)
{

  /*
    F77_NAME(dsyevx,DSYEVX)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  char *name="Eigen_lapack_x";

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
       A[i*n+j] = a[i+1][j+1];
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
  for (i=0;i<EVmax;i++) {
    for (j=0;j<n;j++) {
      /*  a[i+1][j+1]= Z[i*n+j]; */
      a[j+1][i+1]= Z[i*n+j];
    }
  }

  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    ko[i]= ko[i-1];
  }

  /*
  if (INFO>0) {
    printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO);
  }
  */

  if (INFO<0) {
     printf("%s: info=%d\n",name,INFO);
     exit(10);
  }
   
  free(IFAIL); free(IWORK); free(WORK); free(Z); free(A);

}


void Eigen_lapack_d(double **a, double *ko, int n0, int EVmax)
{

  /* 
    F77_NAME(dsyevd,DSYEVD)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  static char *name="Eigen_lapack_d";

  char  *JOBZ="V";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;

  double *A;
  INTEGER LDZ=n;
  INTEGER LWORK,LIWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER INFO;

  int i,j;

  A=(double*)malloc(sizeof(double)*n*n);

  LWORK=  1 + 6*n + 2*n*n;
  WORK=(double*)malloc(sizeof(double)*LWORK);

  LIWORK = 3 + 5*n;
  IWORK=(INTEGER*)malloc(sizeof(INTEGER)*LIWORK);


  IL = 1;
  IU = EVmax; 
 
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

  F77_NAME(dsyevd,DSYEVD)( JOBZ, UPLO, &n, A, &LDA, ko, WORK, &LWORK, IWORK, &LIWORK, &INFO ); 

  /* store eigenvectors */
  for (i=0;i<EVmax;i++) {
    for (j=0;j<n;j++) {
      /* a[i+1][j+1]= Z[i*n+j]; */
      a[j+1][i+1]= A[i*n+j];
    }
  }

  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    ko[i]= ko[i-1];
  }

  /*
  if (INFO>0) {
     printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO);
  }
  */

  if (INFO<0) {
     printf("%s: info=%d\n",name,INFO);
     exit(10);
  }
   
  free(IWORK); free(WORK); free(A);
}


void Eigen_lapack_r(double **a, double *ko, int n0, int EVmax)
{

  /* 
    F77_NAME(dsyevr,DSYEVR)()
  
    input:  n;
    input:  a[n][n];  matrix A
    output: a[n][n];  eigevectors
    output: ko[n];    eigenvalues 
  */
    
  static char *name="Eigen_lapack_r";

  char  *JOBZ="V";
  char  *RANGE="I";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;

  double *A,*Z;
  INTEGER LDZ=n;
  INTEGER LWORK,LIWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *ISUPPZ; 
  INTEGER INFO;

  int i,j;

  A=(double*)malloc(sizeof(double)*n*n);
  Z=(double*)malloc(sizeof(double)*n*n);

  LWORK= (n+16)*n;   /*  n*26 of (n+6)*n */
  WORK=(double*)malloc(sizeof(double)*LWORK);

  LIWORK = (n+1)*n;  /*  n*10 or ??? */
  IWORK=(INTEGER*)malloc(sizeof(INTEGER)*LIWORK);

  ISUPPZ =(INTEGER*)malloc(sizeof(INTEGER)*n*2);

  IL = 1;
  IU = EVmax; 
 
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

  F77_NAME(dsyevr,DSYEVR)( JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
           &ABSTOL, &M, ko, Z, &LDZ, ISUPPZ, WORK, &LWORK,
           IWORK, &LIWORK, &INFO ); 

  /* store eigenvectors */
  for (i=0;i<EVmax;i++) {
    for (j=0;j<n;j++) {
      /*  a[i+1][j+1]= Z[i*n+j]; */
      a[j+1][i+1]= Z[i*n+j];
    }
  }

  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    ko[i]= ko[i-1];
  }

  /*
  if (INFO>0) {
     printf("\n%s: error in dsyevr_, info=%d\n\n",name,INFO);
  }
  */

  if (INFO<0) {
     printf("%s: info=%d\n",name,INFO);
     exit(10);
  }
   
  free(ISUPPZ); free(IWORK); free(WORK); free(Z); free(A);

}



#pragma optimization_level 1
void Eigen_HH(double **ac, double *ko, int n, int EVmax)
{
  /**********************************************************************
    Eigen_HH:

    Eigen_HH.c is a subroutine to solve a seqular equation without an
    overlap matrix using Householder method and lapack's dstevx, dstegr, 
    or dstedc.

    Log of Eigen_HH.c:

       Nov/22/2004  Released by T.Ozaki

  ***********************************************************************/

  double ABSTOL=LAPACK_ABSTOL;
  double **ad,*D,*E,
                *b1,*u,*uu,
                *p,*q,*s,*c,
                s1,s2,s3,ss,u1,u2,r,p1,
                s20,s21,s22,s23,
                xsum,bunbo,si,co,sum,
                a1,a2,a3,a4,a5,a6,b7,
                x1,x2,xap,tmp1,tmp2,
                bb,bb1,ui,uj,uij;

  double ss0,ss1,ss2,ss3;
  double r0,r1,r2,r3,p10,p11,p12,p13;
  double tmp10,tmp11,tmp12,tmp13;
  double tmp20,tmp21,tmp22,tmp23;
 
  int jj,jj1,jj2,k,ii,ll,i3,i2,j2,i1s,
             i,j,i1,j1,n1,n2,ik,
             jk,po1,nn,count,ks;

  double Stime, Etime;
  double Stime1, Etime1;
  double Stime2, Etime2;
  double time1,time2;

  /****************************************************
    allocation of arrays:
  ****************************************************/

  n2 = n + 5;

  ad = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    ad[i] = (double*)malloc(sizeof(double)*n2);
  }

  b1 = (double*)malloc(sizeof(double)*n2);
  u = (double*)malloc(sizeof(double)*n2);
  uu = (double*)malloc(sizeof(double)*n2);
  p = (double*)malloc(sizeof(double)*n2);
  q = (double*)malloc(sizeof(double)*n2);
  s = (double*)malloc(sizeof(double)*n2);
  c = (double*)malloc(sizeof(double)*n2);

  D = (double*)malloc(sizeof(double)*n2);
  E = (double*)malloc(sizeof(double)*n2);

  for (i=1; i<=(n+2); i++){
    uu[i] = 0.0;
  }

  if (measure_time==1) printf("size n=%3d EVmax=%2d\n",n,EVmax);
  if (measure_time==1) dtime(&Stime);

  /****************************************************
                   Householder method
  ****************************************************/

  for (i=1; i<=(n-2); i++){

    /* original version */

    /*
    s1 = ac[i+1][i] * ac[i+1][i];
    s2 = 0.0;
    u[i+1] = ac[i+1][i];
    for (i1=i+2; i1<=n; i1++){
      tmp1 = ac[i1][i]; 
      s2 += tmp1*tmp1;
      u[i1] = tmp1;
    }
    s3 = fabs(s1 + s2);
    */

    /* unrolling version */

    s1 = ac[i+1][i] * ac[i+1][i];
    u[i+1] = ac[i+1][i];

    s20 = 0.0;
    s21 = 0.0;
    s22 = 0.0;
    s23 = 0.0;

    for (i1=i+2; i1<=(n-3); i1+=4){

      s20 += ac[i][i1+0]*ac[i][i1+0];
      s21 += ac[i][i1+1]*ac[i][i1+1];
      s22 += ac[i][i1+2]*ac[i][i1+2];
      s23 += ac[i][i1+3]*ac[i][i1+3];

      u[i1+0] = ac[i][i1+0];
      u[i1+1] = ac[i][i1+1];
      u[i1+2] = ac[i][i1+2];
      u[i1+3] = ac[i][i1+3];
    }

    i1s = n + 1 - (n+1-(i+2))%4;

    for (i1=i1s; ((i+2)<=i1 && i1<=n); i1++){
      tmp1 = ac[i][i1]; 
      s20 += tmp1*tmp1;
      u[i1] = tmp1;
    }

    s2 = s20 + s21 + s22 + s23;
    s3 = fabs(s1 + s2);

    if (ABSTOL<fabs(ac[i+1][i])){
      if (ac[i+1][i]<0.0)    s3 =  sqrt(s3);
      else                   s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if (ABSTOL<fabs(s2)){

      ss = ac[i+1][i];
      ac[i+1][i] = s3;
      ac[i][i+1] = s3;
      u[i+1] = u[i+1] - s3;
      u1 = s3 * s3 - ss * s3;
      u2 = 2.0 * u1;
      uu[i] = u2;
      b1[i] = ss - s3;

      /* original version */

      /*
      r = 0.0;
      for (i1=i+1; i1<=n; i1++){
	p1 = 0.0;
	for (j=i+1; j<=n; j++){
	  p1 += ac[i1][j] * u[j];
	}
	p[i1] = p1 / u1;
	r += u[i1] * p[i1];
      }
      r = r / u2;
      */

      /* unrolling version */

      r0 = 0.0;
      r1 = 0.0;
      r2 = 0.0;
      r3 = 0.0;

      for (i1=i+1; i1<=(n-3); i1+=4){

	p10 = 0.0;
	p11 = 0.0;
	p12 = 0.0;
	p13 = 0.0;

	for (j=i+1; j<=n; j++){
	  p10 += ac[i1+0][j] * u[j];
	  p11 += ac[i1+1][j] * u[j];
	  p12 += ac[i1+2][j] * u[j];
	  p13 += ac[i1+3][j] * u[j];
	}

	p[i1+0] = p10 / u1;
	p[i1+1] = p11 / u1;
	p[i1+2] = p12 / u1;
	p[i1+3] = p13 / u1;

	r0 += u[i1+0] * p[i1+0];
	r1 += u[i1+1] * p[i1+1];
	r2 += u[i1+2] * p[i1+2];
	r3 += u[i1+3] * p[i1+3];
      }

      i1s = n + 1 - (n+1-(i+1))%4;

      for (i1=i1s; ((i+1)<=i1 && i1<=n); i1++){
	p1 = 0.0;
	for (j=i+1; j<=n; j++){
	  p1 += ac[i1][j] * u[j];
	}
	p[i1] = p1 / u1;
	r0 += u[i1] * p[i1];
      }

      r = (r0+r1+r2+r3) / u2;

      /* original version */

      /*
      for (i1=i+1; i1<=n; i1++){
	q[i1] = p[i1] - r * u[i1];
      }
      */

      /* unrolling version */

      for (i1=i+1; i1<=(n-3); i1+=4){
	q[i1+0] = p[i1+0] - r * u[i1+0];
	q[i1+1] = p[i1+1] - r * u[i1+1];
	q[i1+2] = p[i1+2] - r * u[i1+2];
	q[i1+3] = p[i1+3] - r * u[i1+3];
      }

      i1s = n + 1 - (n+1-(i+1))%4;

      for (i1=i1s; ((i+1)<=i1 && i1<=n); i1++){
	q[i1] = p[i1] - r * u[i1];
      }

      /* original version */

      /*
      for (i1=i+1; i1<=n; i1++){
        tmp1 = u[i1];
        tmp2 = q[i1]; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1][j1] -= tmp1 * q[j1] + tmp2 * u[j1];
	}
      }
      */

      /* unrolling version */

      for (i1=i+1; i1<=(n-3); i1+=4){

        tmp10 = u[i1+0];
        tmp11 = u[i1+1];
        tmp12 = u[i1+2];
        tmp13 = u[i1+3];

        tmp20 = q[i1+0]; 
        tmp21 = q[i1+1]; 
        tmp22 = q[i1+2]; 
        tmp23 = q[i1+3]; 

	for (j1=i+1; j1<=n; j1++){
	  ac[i1+0][j1] -= tmp10 * q[j1] + tmp20 * u[j1];
	  ac[i1+1][j1] -= tmp11 * q[j1] + tmp21 * u[j1];
	  ac[i1+2][j1] -= tmp12 * q[j1] + tmp22 * u[j1];
	  ac[i1+3][j1] -= tmp13 * q[j1] + tmp23 * u[j1];
	}
      }

      i1s = n + 1 - (n+1-(i+1))%4;

      for (i1=i1s; ((i+1)<=i1 && i1<=n); i1++){
        tmp1 = u[i1];
        tmp2 = q[i1]; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1][j1] -= tmp1 * q[j1] + tmp2 * u[j1];
	}
      }

    }
  }

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j] = ac[i][j];
    }
  }

  /*
  for (i=1; i<=n; i++){
    printf("i=%4d  D=%18.15f E=%18.15f\n",i,ad[i][i],ad[i][i+1]);
  }
  */

  if (measure_time==1){
    dtime(&Etime);
    printf("T1   %15.12f\n",Etime-Stime);
  }

  /****************************************************
                  call a lapack routine
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (i=1; i<=n; i++){
    D[i-1] = ad[i][i];
    E[i-1] = ad[i][i+1];
  }

  if      (dste_flag==0) lapack_dstegr1(n,EVmax,D,E,ko,ac);
  else if (dste_flag==1) lapack_dstedc1(n,D,E,ko,ac);
  else if (dste_flag==2) lapack_dstevx1(n,EVmax,D,E,ko,ac);

  if (measure_time==1){
    dtime(&Etime);
    printf("T2   %15.12f\n",Etime-Stime);
  }

  /****************************************************
    transformation of eigenvectors to original space
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (i=2; i<=n-1; i++){
    ad[i-1][i] = b1[i-1];
  }

  /* original version */

  /*
  for (k=1; k<=EVmax; k++){
    for (nn=2; nn<=n-1; nn++){
      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){
	ss = 0.0;
	for (i=n-nn+1; i<=n; i++){
	  ss += ad[n-nn][i] * ac[k][i];
	}
	ss = 2.0*ss/uu[n-nn];
	for (i=n-nn+1; i<=n; i++){
	  ac[k][i] -= ss * ad[n-nn][i];
	}
      }
    }
  }
  */

  /* unrolling version */

  for (k=1; k<=(EVmax-3); k+=4){

    for (nn=2; nn<=n-1; nn++){
      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	ss0 = 0.0;
	ss1 = 0.0;
	ss2 = 0.0;
	ss3 = 0.0;

	for (i=n-nn+1; i<=n; i++){
	  ss0 += ad[n-nn][i] * ac[k+0][i];
	  ss1 += ad[n-nn][i] * ac[k+1][i];
	  ss2 += ad[n-nn][i] * ac[k+2][i];
	  ss3 += ad[n-nn][i] * ac[k+3][i];
	}

	ss0 = 2.0*ss0/uu[n-nn];
	ss1 = 2.0*ss1/uu[n-nn];
	ss2 = 2.0*ss2/uu[n-nn];
	ss3 = 2.0*ss3/uu[n-nn];

	for (i=n-nn+1; i<=n; i++){
	  ac[k+0][i] -= ss0 * ad[n-nn][i];
	  ac[k+1][i] -= ss1 * ad[n-nn][i];
	  ac[k+2][i] -= ss2 * ad[n-nn][i];
	  ac[k+3][i] -= ss3 * ad[n-nn][i];
	}

      }
    }
  }

  ks = EVmax - EVmax%4 + 1;

  for (k=ks; k<=EVmax; k++){
    for (nn=2; nn<=n-1; nn++){
      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){
	ss = 0.0;
	for (i=n-nn+1; i<=n; i++){
	  ss += ad[n-nn][i] * ac[k][i];
	}
	ss = 2.0*ss/uu[n-nn];
	for (i=n-nn+1; i<=n; i++){
	  ac[k][i] -= ss * ad[n-nn][i];
	}
      }
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T4   %15.12f\n",Etime-Stime);
  }

  /****************************************************
                     normalization
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (j=1; j<=EVmax; j++){
    sum = 0.0;
    for (i=1; i<=n; i++){
      sum = sum + ac[j][i] * ac[j][i];
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      ac[j][i] = ac[j][i] * sum;
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T5   %15.12f\n",Etime-Stime);
  }

  /****************************************************
            Eigenvectors to the "ac" array
  ****************************************************/

  for (i=1; i<=n; i++){
    for (j=(i+1); j<=n; j++){
      tmp1 = ac[i][j];
      tmp2 = ac[j][i];
      ac[i][j] = tmp2;
      ac[j][i] = tmp1;
    }
  }

  /****************************************************
                  freeing of arrays:
  ****************************************************/

  for (i=0; i<n2; i++){
    free(ad[i]);
  }
  free(ad);

  free(b1);
  free(u);
  free(uu);
  free(p);
  free(q);
  free(s);
  free(c);
  free(D);
  free(E);
}






void Eigen_HHQR(double **ac, double *ko, int n, int EVmax)
{
  /**********************************************************************
    Eigen_HHQR:

    Eigen_HHQR.c is a subroutine to solve a seqular equation without an
    overlap matrix using Householder-QR method.

    Log of Eigen_HHQR.c:

       Nov/22/2004  Released by T.Ozaki

  ***********************************************************************/

  double ABSTOL=LAPACK_ABSTOL;

  double **ad,**aq,**b,
                *b1,*u,*uu,
                *p,*q,*s,*c,*ko1,
                s1,s2,s3,ss,u1,u2,r,p1,
                xsum,bunbo,si,co,sum,
                a1,a2,a3,a4,a5,a6,b7,r1,r2,
                r3,x1,x2,xap,tmp1,tmp2,
                bb,bb1,ui,uj,uij;

  int jj,jj1,jj2,k,ii,ll,i3,i2,j2,
             *jun,*shuku,*po,
             i,j,i1,j1,n1,n2,ik,jk,po1,nn,count;

  double Stime, Etime;
  double Stime1, Etime1;
  double Stime2, Etime2;
  double time1,time2;

  /****************************************************
    allocation of arrays:
  ****************************************************/

  n2 = n + 5;

  jun = (int*)malloc(sizeof(int)*n2);
  shuku = (int*)malloc(sizeof(int)*n2);
  po = (int*)malloc(sizeof(int)*n2);

  ad = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    ad[i] = (double*)malloc(sizeof(double)*n2);
  }

  aq = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    aq[i] = (double*)malloc(sizeof(double)*n2);
  }

  b = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    b[i] = (double*)malloc(sizeof(double)*n2);
  }

  b1 = (double*)malloc(sizeof(double)*n2);
  u = (double*)malloc(sizeof(double)*n2);
  uu = (double*)malloc(sizeof(double)*n2);
  p = (double*)malloc(sizeof(double)*n2);
  q = (double*)malloc(sizeof(double)*n2);
  s = (double*)malloc(sizeof(double)*n2);
  c = (double*)malloc(sizeof(double)*n2);
  ko1 = (double*)malloc(sizeof(double)*n2);


  for (i=1; i<=(n+2); i++){
    uu[i] = 0.0;
    ko1[i] = 1.0e+10;
  }
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j] = 0.0;
      aq[i][j] = ac[i][j];
    }
  }

  dtime(&Stime);

  /****************************************************
                    Householder method
  ****************************************************/

  for (i=1; i<=(n-2); i++){

    s1 = ac[i+1][i] * ac[i+1][i];
    s2 = 0.0;
    u[i+1] = ac[i+1][i];
    for (i1=i+2; i1<=n; i1++){
      tmp1 = ac[i1][i]; 
      s2 += tmp1*tmp1;
      u[i1] = tmp1;
    }
    s3 = fabs(s1 + s2);

    if (ABSTOL<fabs(ac[i+1][i])){
      if (ac[i+1][i]<0.0)    s3 =  sqrt(s3);
      else                   s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if (ABSTOL<fabs(s2)){

      ss = ac[i+1][i];
      ac[i+1][i] = s3;
      ac[i][i+1] = s3;
      u[i+1] = u[i+1] - s3;
      u1 = s3 * s3 - ss * s3;
      u2 = 2.0 * u1;
      uu[i] = u2;
      b1[i] = ss - s3;

      r = 0.0;
      for (i1=i+1; i1<=n; i1++){
	p1 = 0.0;
	for (j=i+1; j<=n; j++){
	  p1 += ac[i1][j] * u[j];
	}
	p[i1] = p1 / u1;
	r += u[i1] * p[i1];
      }
      r = r / u2;

      for (i1=i+1; i1<=n; i1++){
	q[i1] = p[i1] - r * u[i1];
      }

      for (i1=i+1; i1<=n; i1++){
        tmp1 = u[i1];
        tmp2 = q[i1]; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1][j1] -= tmp1 * q[j1] + tmp2 * u[j1];
	}
      }

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  printf("%15.12f ",ac[i1][j1]);
	}
	printf("\n");
      }
      exit(0);
      */


    }
  }



  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j] = ac[i][j];
      ac[i][j] = 0.0;
    }
  }
  for (i=1; i<=n; i++){
    for (j=i-1; j<=i+1; j++){
      ac[i][j] = ad[i][j];
    }
  }


  dtime(&Etime);

  printf("T1 %15.12f\n",Etime-Stime);

  /****************************************************
                       QR method
  ****************************************************/

  dtime(&Stime);

  count = 0;
  xsum = 0.0;
  n1 = n;

  do {

    for (i=1; i<=(n1-1); i++){

      if ( ABSTOL<=fabs(ac[i+1][i]) ){

	bunbo = sqrt(ac[i][i] * ac[i][i] + ac[i+1][i] * ac[i+1][i]);
	si = ac[i+1][i] / bunbo;
	co = ac[i  ][i] / bunbo;

	s[i] = si;
	c[i] = co;
	ac[i  ][i] = bunbo;
	ac[i+1][i] = 0.0;

	a1 = ac[i  ][i+1];
	a2 = ac[i+1][i+1];
	ac[i  ][i+1] =  a1 * co + a2 * si;
	ac[i+1][i+1] = -a1 * si + a2 * co;

	a3 = ac[i  ][i+2];
	a4 = ac[i+1][i+2];
	ac[i  ][i+2] =  a3 * co + a4 * si;
	ac[i+1][i+2] = -a3 * si + a4 * co;

	po[i] = 1;
      }
      else
	po[i] = 0;
    }

    for (j=1; j<=(n1-1); j++){
      if (po[j]==1){
	si = s[j];
	co = c[j];

	a1 = ac[j-1][j  ];
	a2 = ac[j-1][j+1];

	ac[j-1][j  ] = a1 * co + a2 * si;
	ac[j-1][j+1] = 0.0;

	a3 = ac[j][j  ];
	a4 = ac[j][j+1];
	ac[j][j  ] =  a3 * co + a4 * si;
	ac[j][j+1] = -a3 * si + a4 * co;

	a5 = ac[j+1][j  ];
	a6 = ac[j+1][j+1];

	ac[j+1][j  ] =  a5 * co + a6 * si;
	ac[j+1][j+1] = -a5 * si + a6 * co;

      }
    }

    r1 = ac[n1-1][n1-1] + ac[n1][n1];
    r2 = ac[n1-1][n1-1] - ac[n1][n1];
    r3 = r2 * r2 + 4.0 * ac[n1-1][n1] * ac[n1][n1-1];
    if (0.0<=r3)
      r3 = sqrt(r3);
    else 
      r3 = 0.0;

    x1 = (r1 + r3) * 0.5000;
    x2 = (r1 - r3) * 0.5000;
    if (fabs(x1)>fabs(x2))
      xap = x1;
    else
      xap = x2;
    for (i1=1; i1<=n1; i1++){
      ac[i1][i1] = ac[i1][i1]- xap;
    }
    xsum = xsum + xap;

    /****************************************************
       Eigenvalues sensitively depend on the criterion.
    ****************************************************/

    if (fabs(ac[n1][n1-1])<1.0e-4 || 30<count){
      ko[n1] = ac[n1][n1] + xsum;
      n1--;
      count = 0;
    }
    else {
      count++;
    }

  }
  while(n1 >= 1);

  dtime(&Etime);
  printf("T2 %15.12f\n",Etime-Stime);

  ko[1] = ac[1][1] + xsum;

  for (ik=1; ik<=n; ik++){
    po1 = 0;
    jj = 1;
    do {
      if (ko[ik]<ko1[jj]){
	for (jk=n; jk>=jj+1; jk--){
	  ko1[jk] = ko1[jk-1];
	}
	ko1[jj] = ko[ik];
	po1 = 1;
      }
      jj++;
    }
    while(po1==0 && jj<=n);
  }

  for (i=1; i<=n; i++){
    shuku[i] = 0;
  }
  for (i=1; i<=n; i++){
    ko[i] = ko1[i];
  }
  for (i=1; i<=n; i++){
    j = i;
    po1 = 0;
    do {

      /****************************************************
        Eigenvalues sensitively depend on the criterion.
        Especially, in case of degeneracy, some problem
        appears. We choose 0.0000001 for the numerical
        stability.
      ****************************************************/

      if (fabs(ko[j]-ko[j+1])<.000001)
	j = j + 1;
      else
	po1 = 1;
    }
    while(po1==0);

    if (j!=i) {
      for (k=i; k<=j; k++){
	shuku[k] = j - i;
      }
      i = j;
    }
  }




  dtime(&Stime);
  time1 = 0.0;
  time2 = 0.0;

  /****************************************************
    Calculate eigenvectors using the iterative method
    combined with Gauss method.

           First, the lower part is calculated.
  ****************************************************/

  k = 1;
  do{

    if (shuku[k]==0){

  dtime(&Stime1);

  /*
      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  ac[i][j] = 0.0;
	}
      }
  */

      for (ii=1; ii<=n; ii++){
	b[k][ii] = .50;
      }

  dtime(&Etime1);
  time1 += Etime1-Stime1; 

  dtime(&Stime2);

      for (ll=1; ll<=2; ll++){

        /****************************************************
                 First, the lower part is calculated.
        ****************************************************/

	for (i=1; i<=n; i++){
	  for (j=i-1; j<=i+1;j++){
	    if (j>0 && j<n+1){
	      ac[i][j] = ad[i][j];
	    }
	  }
	}
	for (i=1; i<=n; i++){
	  ac[i][i] = ac[i][i] - ko[k] + .00000001;
	}

	for (i=1; i<=n-1; i++){

	  if (fabs(ac[i][i])<1.0e-50){
	    printf("division by Zero! \n");
	    for (i1=i+1; i1<=i+1; i1++){
              if (ABSTOL<fabs(ac[i1][i])){

		a1 = b[k][i];
		a2 = b[k][i1];
		b[k][i]  = a2;
	        b[k][i1] = a1;

		for (j1=1; j1<=n; j1++){
		  a1 = ac[i ][j1];
		  a2 = ac[i1][j1];
		  ac[i ][j1] = a2;
		  ac[i1][j1] = a1;
		}

	      }
	    }
	  }

          if (ABSTOL<fabs(ac[i+1][i])){
	    bb = ac[i+1][i] / ac[i][i];
	    ac[i+1][i] = 0.0;
	    ac[i+1][i+1] = ac[i+1][i+1] - bb * ac[i][i+1];
	    b[k][i+1] -= bb * b[k][i];
	  }

	}

        /****************************************************
                Second, the upper part is calculated.
        ****************************************************/

	b[k][n] = b[k][n] / ac[n][n];
	for (i=n-1; i>=1; i--){
	  sum = b[k][i+1] * ac[i][i+1];
	  b[k][i] = (b[k][i] - sum) / ac[i][i];
	}
      }

  dtime(&Etime2);
  time2 += Etime2-Stime2; 

      k = k + 1;
    }

    else {

      printf("k=%2d\n",k);

      /****************************************************
                       In case of degeracy
      ****************************************************/

      for (i=1; i<=n; i++){
	jun[i] = i;
      }
      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  ac[i][j] = aq[i][j];
	  if (i==j)
	    ac[i][j] = ac[i][j] - ko[k];
	}
      }
      for (i=1; i<=(n-(shuku[k]+1)); i++){
	s3 = fabs(ac[i][i]);
	i3 = i;
	for (i1=i+1; i1<=n; i1++){
	  if (fabs(ac[i1][i])>s3){
	    s3 = fabs(ac[i1][i]);
	    i3 = i1;
	  }
	}
	if (i3!=i){
	  for (j1=i; j1<=n; j1++){
	    a1 = ac[i][j1];
	    a2 = ac[i3][j1];
	    ac[i][j1] = a2;
	    ac[i3][j1] = a1;
	  }
	}
	if (fabs(ac[i][i])<.00001){
	  s3 = fabs(ac[i][i]);
	  i2 = i;
	  j2 = i;
	  for (i1=i; i1<=n; i1++){
	    for (j1=i+1;j1<=n; j1++){
	      if (fabs(ac[i1][j1])>s3){
		s3 = fabs(ac[i1][j1]);
		i2 = i1;
		j2 = j1;
	      }
	    }
	  }
	  if (i2!=i){
	    for (j1=i; j1<=n; j1++){
	      a1 = ac[i][j1];
	      a2 = ac[i2][j1];
	      ac[i][j1] = a2;
	      ac[i2][j1] = a1;
	    }
	  }
	  if (j2!=i){
	    for (i1=1; i1<=n; i1++){
	      a1 = ac[i1][i];
	      a2 = ac[i1][j2];
	      ac[i1][i] = a2;
	      ac[i1][j2] = a1;
	    }
	    jj1 = jun[j2];
	    jj2 = jun[i];
	    jun[i] = jj1;
	    jun[j2] = jj2;
	  }
	}
	bb = ac[i][i];
	for (j1=1; j1<=n; j1++){
	  ac[i][j1] = ac[i][j1] / bb;
	}
	for (i1=1; i1<=n; i1++){
	  if (i1!=i){
            if (0.000000000001<fabs(ac[i1][i])){
	      bb = ac[i1][i];
	      for (j=i; j<=n; j++){
		ac[i1][j] = ac[i1][j] - bb * ac[i][j];
	      }
	    }
	  }
	}
      }
      for (j1=1; j1<=(shuku[k]+1); j1++){
	for (i1=1; i1<=(n-(shuku[k]+1)); i1++){
	  b[k+j1-1][jun[i1]] = 0.0;
	  for (j2=1; j2<=j1; j2++){
	    b[k+j1-1][jun[i1]] += ac[i1][n-j2+1];
	  }
	}
      }
      for (i1=0; i1<=shuku[k]; i1++){
	for (j1=0; j1<=shuku[k]; j1++){
	  if (j1<=i1)
	    b[k+i1][jun[n-j1]] = -1.0;
	  else
	    b[k+i1][jun[n-j1]] = 0.0;
	}
      }
      k = k + shuku[k] + 1;
    }
  }
  while(k<=n);

  dtime(&Etime);

  printf("T3 %15.12f\n",Etime-Stime);
  printf("time1 %15.12f\n",time1);
  printf("time2 %15.12f\n",time2);


  for (j=1; j<=n; j++){
    sum = 0.0;
    for (i=1; i<=n; i++){
      sum = sum + b[j][i] * b[j][i];
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      b[j][i] = b[j][i] * sum;
    }
  }

  printf("\n\n b\n");
  for (j=1; j<=n; j++){
    for (i=1; i<=n; i++){
      printf("%10.7f ",b[i][j]);
    }
    printf("\n");
  }



  /****************************************************
     Transformation of eigenvectors to original space
  ****************************************************/

  dtime(&Stime);

  for (i=2; i<=n-1; i++){
    ad[i-1][i] = b1[i-1];
  }

  for (k=1; k<=n; k++){
    if (shuku[k]==0){
      for (nn=2; nn<=n-1; nn++){
        if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){
	  ss = 0.0;
	  for (i=n-nn+1; i<=n; i++){
	    ss += ad[n-nn][i] * b[k][i];
	  }
	  ss = 2.0*ss/uu[n-nn];
	  for (i=n-nn+1; i<=n; i++){
	    b[k][i] -= ss * ad[n-nn][i];
	  }
	}
      }
    }
  }

  dtime(&Etime);
  printf("T4 %15.12f\n",Etime-Stime);

  /****************************************************
                   Gram-Schmidt method
  ****************************************************/

  dtime(&Stime);

  for (j=1; j<=n; j++){
    if (shuku[j]==0){
      sum = 0.0;
      for (i=1; i<=n; i++){
	sum = sum + b[j][i] * b[j][i];
      }
      sum = 1.0/sqrt(sum);
      for (i=1; i<=n; i++){
	b[j][i] = b[j][i] * sum;
      }
    }
  }
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ac[i][j] = 0.0;
    }
  }
  for (i=1; i<=n; i++){
    if (shuku[i]!=0){
      for (j=1; j<=n; j++){
	if (j!=i && shuku[j]==0){
	  sum = 0.0;
	  for (i1=1; i1<=n; i1++){
	    sum = sum + b[i][i1] * b[j][i1];
	  }
	  for (i1=1; i1<=n; i1++){
	    ac[i1][i] = ac[i1][i] + sum * b[j][i1];
	  }
	}
      }
      for (i1=1; i1<=n; i1++){
	b[i][i1] = b[i][i1] - ac[i1][i];
      }
      sum = 0.0;
      for (i1 = 1; i1<=n; i1++){
	sum = sum + b[i][i1] * b[i][i1];
      }
      sum = 1.0/sqrt(sum);
      for (i1=1; i1<=n; i1++){
	b[i][i1] = b[i][i1] * sum;
      }
      shuku[i] = 0;
    }
  }

  dtime(&Etime);
  printf("T5 %15.12f\n",Etime-Stime);

  /****************************************************
            Eigen vectors to the "ac" array
  ****************************************************/

  for (j=1; j<=n; j++){
    for (i=1; i<=n; i++){
      ac[i][j] = b[j][i];
    }
  }

  /****************************************************
                  freeing of arrays:
  ****************************************************/

  free(jun);
  free(shuku);
  free(po);

  for (i=0; i<n2; i++){
    free(ad[i]);
  }
  free(ad);

  for (i=0; i<n2; i++){
    free(aq[i]);
  }
  free(aq);

  for (i=0; i<n2; i++){
    free(b[i]);
  }
  free(b);

  free(b1);
  free(u);
  free(uu);
  free(p);
  free(q);
  free(s);
  free(c);
  free(ko1);

}

