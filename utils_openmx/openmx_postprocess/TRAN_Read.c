#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include "tran_prototypes.h"

#define LEN  256

void TRAN_Read_double(char *str, int n1,int n2, double *a)
#define a_ref(i,j) a[n1*((j)-1)+(i)-1]
{

  int i,j;
  int m1,m2,d1,d2;
   double x1,x2;
  FILE *fp;
  char buf[LEN];

  if  (   (fp=fopen(str,"r"))==NULL ) {
    printf("cannot open %s and continue\n", str);
    return;
  }

  printf("TRAN_Read_double> Read data from flle=%s\n",str);
  
  /* read header */
  fgets(buf, LEN,fp);
  sscanf(buf,"%d %d %d %d",&m1,&m2,&d1,&d2);

  /* check consistency of the size */
  if ( m1=!n1  || m2!=n2 ) {
     printf("\n\nERROR, size is different, %s , %d %d %d %d\n\n\n",str, n1,n2,m1,m2);
     goto last;   
  }

  /* initialize */
  for (i=0;i<n1*n2; i++) {
     a[i]=0.0;
  }

  /* read and set */
  while (1) {
    if (fgets(buf, LEN,fp)==NULL) { goto last; }
    if ( sscanf(buf,"%d %d %lf %lf", &i,&j, &x1,&x2)==0 ) { goto last; }
    a_ref(i,j) = x1;   
  }

  /* finally close the file */
last:
  fclose(fp);
}

void TRAN_Read_dcomplex(char *str, int n1,int n2, dcomplex *a)
#define a_ref(i,j) a[n1*((j)-1)+(i)-1]
{

  int i,j;
  int m1,m2,d1,d2;
   double x1,x2;
  FILE *fp;
  char buf[LEN];

  if  (   (fp=fopen(str,"r"))==NULL ) {
    printf("cannot open %s and continue\n", str);
    return;
  }

  printf("TRAN_Read_dcomplex> Read data from flle=%s\n",str);

  /* read header */
  fgets(buf, LEN,fp);
  sscanf(buf,"%d %d %d %d",&m1,&m2,&d1,&d2);

  /* check consistency of the size */
  if ( m1=!n1  || m2!=n2 ) {
     printf("\n\nERROR, size is different, %s , %d %d %d %d\n\n\n",str, n1,n2,m1,m2);
     goto last;
  }

  /* initialize */
  for (i=0;i<n1*n2; i++) {
     a[i].r=0.0;
     a[i].i=0.0;
  }

  /* read and set */
  while (1) {
    if (fgets(buf, LEN,fp)==NULL) { goto last; }
    if ( sscanf(buf,"%d %d %lf %lf", &i,&j, &x1,&x2)==0 ) { goto last; }
    a_ref(i,j).r = x1;
    a_ref(i,j).i = x2;
  }

  /* finally close the file */
last:
  fclose(fp);


}



void TRAN_FRead2_binary_double(FILE *fp, int n1, int n2, double *gc)
#define gc_ref(i,j) gc[ n1*(j)+i ]
{

   int i,j;
   double *v;
   int iv[10];
   int i1,i2;

   printf("n=%d %d\n",n1,n2);

   v=(double*)malloc(sizeof(double)*n1);

   fread(iv,sizeof(int),2,fp);
   i1=iv[0];
   i2=iv[1];
   if (i1!=n1 || i2!=n2) {
      printf(" i1!=n1 || i2!=n2,  i1=%d i2=%d n1=%d n2=%d\n",i1,i2,n1,n2);
      exit(0);
   }

   for (i=0;i<n1*n2;i++) gc[i]=0.0;

   for (j=0;j<n2;j++) {
      fread(iv,sizeof(int),3,fp);
      if (iv[0]!=j) {
        printf("data format error j=%d iv[0]=%d\n",j,iv[0]);
        exit(0); 
      }
      i1=iv[1];
      i2=iv[2];
/*      printf("%d of %d i=[%d:%d]\n",j,n2,i1,i2); */
      if ( i1<=i2) {
      fread(v,sizeof(double),i2-i1+1,fp);
      for (i=i1;i<=i2;i++) {
         gc_ref(i,j) = v[i-i1];
      }
      }
   }

  free(v);

}


