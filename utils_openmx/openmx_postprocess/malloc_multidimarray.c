#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "openmx_common.h"

#ifdef DEBUG
#undef DEBUG
#endif

void *malloc_multidimarray(char *type, int N, int *size)
{
   int i;
   void **ret,*r;

   if (strcasecmp(type,"double")==0) {
      if (N==1) {
         r=  (void*)malloc(sizeof(double)*size[0]);
#ifdef DEBUG
         printf("malloc_multidimarray: %x  (size=%d)\n",r,size[0]);
#endif
         return r;
      }
      else {
          ret=(void**)malloc(sizeof(double*)*size[0]);
#ifdef DEBUG
          printf("malloc_multidimarray: %x\n next size=%d\n",ret,size[0]);
#endif
          for (i=0;i<size[0];i++) {
              ret[i]=(void*)malloc_multidimarray(type,N-1,&size[1]);
          }
          return ret;
      }
  }
  else if (strcasecmp(type,"dcomplex")==0) {
      if (N==1) {
          r=(void*)malloc(sizeof(dcomplex)*size[0]);
#ifdef DEBUG
          printf("malloc_multidimarray: %x  (size=%d)\n",r,size[0]);
#endif
          return r;
      }
      else {
          ret=(void**)malloc(sizeof(dcomplex*)*size[0]);
#ifdef DEBUG
          printf("malloc_multidimarray: %x\n next size=%d\n",ret,size[0]);
#endif
          for (i=0;i<size[0];i++) {
              ret[i]=(void*)malloc_multidimarray(type,N-1,&size[1]);
          }
          return ret;
      }
  }
  else if (strcasecmp(type,"float")==0) {
      if (N==1) {

          return (void*)malloc(sizeof(float)*size[0]);
      }
      else {
          ret=(void**)malloc(sizeof(float*)*size[0]);
          for (i=0;i<size[0];i++) {
              ret[i]=(void*)malloc_multidimarray(type,N-1,&size[1]);
          }
          return ret;
      }
  }
  else if (strcasecmp(type,"int")==0) {
      if (N==1) {

          return (void*)malloc(sizeof(int)*size[0]);
      }
      else {
          ret=(void**)malloc(sizeof(int*)*size[0]);
          for (i=0;i<size[0];i++) {
              ret[i]=(void*)malloc_multidimarray(type,N-1,&size[1]);
          }
          return ret;
      }
  }
  else if (strcasecmp(type,"char")==0) {
      if (N==1) {

          return (void*)malloc(sizeof(char)*size[0]);
      }
      else {
          ret=(void**)malloc(sizeof(char*)*size[0]);
          for (i=0;i<size[0];i++) {
              ret[i]=(void*)malloc_multidimarray(type,N-1,&size[1]);
          }
          return ret;
      }
  }
  else {
      printf("not supported type=%s\n",type);
      exit(10);
  }
}


void free_multidimarray(void **p,  int N, int *size)
{
   int i;

   if (N>1) {
     for (i=(size[0]-1);i>=0;i--){
       free_multidimarray(p[i],N-1,&size[1]);
       p[i]=NULL;
     }
   }
#if DEBUG
   printf("free_multidimarray: %x \n",p);
#endif
   free(p);

}

#if 0
main()
{
   int N, size[10];
   double *v1,***v3;
   int i,j,k;

   N=1;
   size[0]=5;

   N=3;
   size[1]=2;
   size[2]=3; 
   v3=(double***)malloc_multidimarray("double",N,size);
   for (i=1;i<5;i++) for (j=1;j<2;j++) for(k=1;k<3;k++)  v3[i][j][k]=100*i+10*j+k;
   for (i=1;i<5;i++) for (j=1;j<2;j++) for(k=1;k<3;k++)  printf("%lf\n",v3[i][j][k]);

  free_multidimarray(v3,N,size);


}
#endif
