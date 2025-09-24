/**********************************************************************
    Eigen_lapack.c:

    Eigen_lapack.c is a subroutine to solve a seqular equation without an
    overlap matrix using lapack's routines.

    Log of Eigen_lapack.c:

       10/Dec/2002  Released by H.Kino
       22/Nov/2004  Modified by T.Ozaki
       18/Apr/2013  Modified by A.M.Ito

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define  measure_time   0

static void Eigen_HH2(double *ac, int csize, double *ko, int n, int EVmax);
static int Eigen_lapack_d2(double *a, int csize, double *ko, int n, int EVmax);
static int Eigen_lapack_x2(double *a, int csize, double *ko, int n, int EVmax);

void Eigen_lapack2(double *a, int csize, double *ko, int n, int EVmax)
{
  int info;

#ifdef kcomp
  Eigen_HH2(a, csize, ko, n, EVmax);
#else 
  info = Eigen_lapack_x2(a, csize, ko, n, EVmax);
#endif      


  /*
  if (dste_flag==2){
    info = Eigen_lapack_x2(a, csize, ko, n, EVmax);
    if(info != 0)  Eigen_HH2(a, csize, ko, n, EVmax); 
  }
  else if (dste_flag==1){
    info = Eigen_lapack_d2(a, csize, ko, n, EVmax);
    if(info != 0)  Eigen_HH2(a, csize, ko, n, EVmax); 
  }
  else {
    Eigen_HH2(a, csize, ko, n, EVmax);
  }
  */
} 


#pragma optimization_level 1
void Eigen_HH2(double *ac, int csize, double *ko, int n, int EVmax)
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

    s1 = ac[(i+1)*csize+i] * ac[(i+1)*csize+i];
    u[i+1] = ac[(i+1)*csize+i];

    s20 = 0.0;
    s21 = 0.0;
    s22 = 0.0;
    s23 = 0.0;

    for (i1=i+2; i1<=(n-3); i1+=4){

      s20 += ac[i*csize+i1+0]*ac[i*csize+i1+0];
      s21 += ac[i*csize+i1+1]*ac[i*csize+i1+1];
      s22 += ac[i*csize+i1+2]*ac[i*csize+i1+2];
      s23 += ac[i*csize+i1+3]*ac[i*csize+i1+3];

      u[i1+0] = ac[i*csize+i1+0];
      u[i1+1] = ac[i*csize+i1+1];
      u[i1+2] = ac[i*csize+i1+2];
      u[i1+3] = ac[i*csize+i1+3];
    }

    i1s = n + 1 - (n+1-(i+2))%4;

    for (i1=i1s; ((i+2)<=i1 && i1<=n); i1++){
      tmp1 = ac[i*csize+i1]; 
      s20 += tmp1*tmp1;
      u[i1] = tmp1;
    }

    s2 = s20 + s21 + s22 + s23;
    s3 = fabs(s1 + s2);

    if (ABSTOL<fabs(ac[(i+1)*csize+i])){
      if (ac[(i+1)*csize+i]<0.0)    s3 =  sqrt(s3);
      else                          s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if (ABSTOL<fabs(s2)){

      ss = ac[(i+1)*csize+i];
      ac[(i+1)*csize+i] = s3;
      ac[i*csize+i+1] = s3;
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
	  p10 += ac[(i1+0)*csize+j] * u[j];
	  p11 += ac[(i1+1)*csize+j] * u[j];
	  p12 += ac[(i1+2)*csize+j] * u[j];
	  p13 += ac[(i1+3)*csize+j] * u[j];
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
	  p1 += ac[i1*csize+j] * u[j];
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
	  ac[(i1+0)*csize+j1] -= tmp10 * q[j1] + tmp20 * u[j1];
	  ac[(i1+1)*csize+j1] -= tmp11 * q[j1] + tmp21 * u[j1];
	  ac[(i1+2)*csize+j1] -= tmp12 * q[j1] + tmp22 * u[j1];
	  ac[(i1+3)*csize+j1] -= tmp13 * q[j1] + tmp23 * u[j1];
	}
      }

      i1s = n + 1 - (n+1-(i+1))%4;

      for (i1=i1s; ((i+1)<=i1 && i1<=n); i1++){
        tmp1 = u[i1];
        tmp2 = q[i1]; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1*csize+j1] -= tmp1 * q[j1] + tmp2 * u[j1];
	}
      }

    }
  }

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j] = ac[i*csize+j];
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

  if      (dste_flag==0) lapack_dstegr3(n,EVmax,D,E,ko,ac,csize);
  else if (dste_flag==1) lapack_dstedc3(n,D,E,ko,ac,csize);
  else if (dste_flag==2) lapack_dstevx3(n,EVmax,D,E,ko,ac,csize);

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
	  ss0 += ad[n-nn][i] * ac[(k+0)*csize+i];
	  ss1 += ad[n-nn][i] * ac[(k+1)*csize+i];
	  ss2 += ad[n-nn][i] * ac[(k+2)*csize+i];
	  ss3 += ad[n-nn][i] * ac[(k+3)*csize+i];
	}

	ss0 = 2.0*ss0/uu[n-nn];
	ss1 = 2.0*ss1/uu[n-nn];
	ss2 = 2.0*ss2/uu[n-nn];
	ss3 = 2.0*ss3/uu[n-nn];

	for (i=n-nn+1; i<=n; i++){
	  ac[(k+0)*csize+i] -= ss0 * ad[n-nn][i];
	  ac[(k+1)*csize+i] -= ss1 * ad[n-nn][i];
	  ac[(k+2)*csize+i] -= ss2 * ad[n-nn][i];
	  ac[(k+3)*csize+i] -= ss3 * ad[n-nn][i];
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
	  ss += ad[n-nn][i] * ac[k*csize+i];
	}
	ss = 2.0*ss/uu[n-nn];
	for (i=n-nn+1; i<=n; i++){
	  ac[k*csize+i] -= ss * ad[n-nn][i];
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
      sum += ac[j*csize+i] * ac[j*csize+i];
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      ad[j][i] = ac[j*csize+i] * sum;
    }
  }

  /****************************************************
            Eigenvectors to the "ac" array
  ****************************************************/

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      ac[i*n+j] = ad[i+1][j+1];  
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





int Eigen_lapack_x2(double *a, int csize, double *ko, int n0, int EVmax)
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
      A[i*n+j] = a[(i+1)*csize + j+1];
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


  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    ko[i]= ko[i-1];
  }

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


int Eigen_lapack_d2(double *a, int csize, double *ko, int n0, int EVmax)
{

  /* 
     F77_NAME(dsyevd,DSYEVD)()
  
     input:  n;
     input:  a[n][n];  matrix A
     output: a[n][n];  eigevectors
     output: ko[n];    eigenvalues 
  */
    
  static char *name="Eigen_lapack";

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
      /* A[i*n+j]= a[(i+1)*csize + j+1]; */
      A[i*n+j]= a[(i+1)*csize + j+1];
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


  /* shift ko by 1 */
  for (i=EVmax; i>=1; i--){
    ko[i]= ko[i-1];
  }

  if (INFO>0) {
    /* printf("\n%s: error in dsyevd_, info=%d\n\n",name,INFO); */
  }
  else if (INFO<0) {
    printf("%s: info=%d\n",name,INFO);
    exit(10);
  }
  else{ /* (INFO==0) */
    /* store eigenvectors */
    for (i=0;i<EVmax;i++) {
      for (j=0;j<n;j++) {
        a[i*n + j]= A[i*n+j];
      }
    }
  }
   
  free(IWORK); free(WORK); free(A);

  return INFO;
}

