#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"

static double eps=1.0e-7;
static int max=500;


void TRAN_Print2_set_eps(double e1)
{
  eps = e1;
  printf("TRAN_Print2 eps changed , eps=%le\n",eps);
}

void TRAN_Print2_set_max(int m1)
{
  max=m1;
  printf("TRAN_Print2 max changed , max=%i\n",max);
}


void TRAN_Print2_dcomplex(char *name, int n1,int n2,dcomplex *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total;

  total=0;
  printf("TRAN_Print2_dcomplex <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j++) {
    for (i=0;i<n1;i++) {
      if ( fabs(gc_ref(i,j).r)>eps || fabs(gc_ref(i,j).i)> eps ) {
	printf("%d %d (%le %le)\n", i,j, gc_ref(i,j).r, gc_ref(i,j).i);
	total++;
	if (total > max) { 
	  printf("# of print >max\n");
	  return;
	}
      }
    }
  }

}


void TRAN_Print2_double(char *name, int n1,int n2,double *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total;

  total=0;
  printf("TRAN_Print2_double <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j++) {
    for (i=0;i<n1;i++) {
      if (fabs(gc_ref(i,j)) > eps) {
	printf("%d %d %d %le\n", i,j,n1*(j)+i, gc_ref(i,j));
	total++;
	if ( total >max ){
	  printf("# of print >max\n");
	  return;
	}

      }
    }
  }

}

void TRAN_Print2_dx_dcomplex(char *name, int n1,int dx1, int n2,int dx2, dcomplex *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total=0;


  printf("TRAN_Print2_dcomplex <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j+=dx2) {
    for (i=0;i<n1;i+=dx1) {
      if ( fabs(gc_ref(i,j).r) > eps || fabs(gc_ref(i,j).i) > eps ) {
	printf("%d %d (%le %le)\n",i,j, gc_ref(i,j).r, gc_ref(i,j).i);
	total++;
	if (total >max) {
	  printf("# of print >max\n");
	  return;
	}
      }
    }
  }

}



void TRAN_Print2_dx_double(char *name, int n1,int dx1,int n2,int dx2,double *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total=0;


  printf("TRAN_Print2_double <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j+=dx2) {
    for (i=0;i<n1;i+=dx1) {
      if (fabs(gc_ref(i,j)) > eps ) {
	printf("%d %d %le\n",i,j, gc_ref(i,j));
	total++;
	if (total >max) { 
	  printf("# of print >max\n");
	  return;
	}
      }
    }
  }

}


void TRAN_FPrint2_double(char *name, int n1,int n2,double *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total;
  static int max=10000;
  FILE *fp;

  total=0;
  if ( (fp=fopen(name,"w"))==NULL) {
    printf("can not open %s\nexit\n",name);
    return;
  }
  fprintf(fp,"TRAN_FPrint2_double <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j++) {
    for (i=0;i<n1;i++) {
      if (fabs(gc_ref(i,j)) > eps) {
	fprintf(fp,"%d %d %le\n", i,j, gc_ref(i,j));
	total++;
	if ( total >max ){
	  fprintf(fp,"# of print >max\n");
	  return;
	}

      }
    }
  }
  fclose(fp);

}

void TRAN_FPrint2_dcomplex(char *name, int n1,int n2,dcomplex *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total;
  static double eps=1.0e-30;
  FILE *fp;

  total=0;
  if ( (fp=fopen(name,"w"))==NULL) {
    printf("can not open %s\nexit\n",name);
    return;
  }
  fprintf(fp,"TRAN_FPrint2_double <%s> n=%d %d\n",name,n1,n2);
  for (j=0;j<n2;j++) {
    for (i=0;i<n1;i++) {
      if (fabs(gc_ref(i,j).r) > eps || fabs(gc_ref(i,j).i) > eps) {
	fprintf(fp,"%d %d %30.15lf %30.15lf\n", i,j, gc_ref(i,j).r, gc_ref(i,j).i );
	total++;

      }
    }
  }
  fclose(fp);

}



void TRAN_FPrint2_binary_double(FILE *fp, int n1,int n2,double *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{
  int i,j;
  int total;
  static double eps=1.0e-12;
  double *v;
  int    iv[10];
  int i0,i1;
  int count;

  v = (double*)malloc(sizeof(double)*n1);

  i=0;
  iv[i++]=n1;
  iv[i++]=n2;
  fwrite(iv,sizeof(int),2,fp);

  for (j=0;j<n2;j++) { 
    i0=0;
    i1=-1;
    for (i=0;i<n1;i++) {
      if (fabs(gc_ref(i,j)) >eps ) {
	i0=i;
	break;
      }
    }
    for (i=n1-1;i>=0;i--) {
      if (fabs(gc_ref(i,j)) >eps ) {
	i1=i;
	break;
      }
    }
    count=0;
    iv[count++]=j;
    iv[count++]=i0;
    iv[count++]=i1;
    fwrite(iv,sizeof(int),count,fp);

#if 0
    printf("%d of %d, i= [%d: %d]\n",j,n2,i0,i1);
#endif

    if (i0<=i1) {
      for (i=i0;i<=i1;i++) {
        v[i-i0]=gc_ref(i,j);
      }
      fwrite(v,sizeof(double), i1-i0+1, fp);
    }

  }


  free(v);

}


