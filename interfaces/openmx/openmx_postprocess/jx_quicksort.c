/**********************************************************************
  QuickSort.c:

     QuickSort.c is a subroutine to quick-sort an array a with
     an array b.

  Log of QuickSort.c:

     08/Dec/2005  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jx_quicksort.h"

//#include <malloc/malloc.h>
//#include <assert.h>

//#include "openmx_common.h"

void qsort_int1(long n, int *a)
{
  qsort(a, n, sizeof(int), (int(*)(const void*, const void*))ilists_cmp);
}

void qsort_int(long n, int *a, int *b)
{
  int i;
  ilists *AB;

  AB = (ilists*)malloc(sizeof(ilists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i+1];
    AB[i].b = b[i+1];
  }

  qsort(AB, n, sizeof(ilists), (int(*)(const void*, const void*))ilists_cmp);

  for (i=0; i<n; i++){
    a[i+1] = AB[i].a;
    b[i+1] = AB[i].b;
  }

  free(AB);
}



void qsort_int3(long n, int *a, int *b, int *c)
{
  int i;
  ilist3 *ABC;

  ABC = (ilist3*)malloc(sizeof(ilist3)*n);

  for (i=0; i<n; i++){
    ABC[i].a = a[i];
    ABC[i].b = b[i];
    ABC[i].c = c[i];
  }

  qsort(ABC, n, sizeof(ilist3), (int(*)(const void*, const void*))ilists_cmp);

  for (i=0; i<n; i++){
    a[i] = ABC[i].a;
    b[i] = ABC[i].b;
    c[i] = ABC[i].c;
  }

  free(ABC);
}

void qsort_double(long n, double *a, double *b)
{
  int i;
  dlists *AB;

  AB = (dlists*)malloc(sizeof(dlists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i+1];
    AB[i].b = b[i+1];
  }

  qsort(AB, n, sizeof(dlists), (int(*)(const void*, const void*))dlists_cmp);

  for (i=0; i<n; i++){
    a[i+1] = AB[i].a;
    b[i+1] = AB[i].b;
  }

  free(AB);
}

void qsort_double_int(long n, double *a, int *b)
{
  int i;
  dilists *AB;

  AB = (dilists*)malloc(sizeof(dilists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i];
    AB[i].b = b[i];
  }

  qsort(AB, n, sizeof(dilists), (int(*)(const void*, const void*))dlists_cmp);

  for (i=0; i<n; i++){
    a[i] = AB[i].a;
    b[i] = AB[i].b;
  }

  free(AB);
}

void qsort_double_int2(long n, double *a, int *b)
{
  int i;
  dilists *AB;

  AB = (dilists*)malloc(sizeof(dilists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i];
    AB[i].b = b[i];
  }

  qsort(AB, n, sizeof(dilists), (int(*)(const void*, const void*))dlists_cmp2);

  for (i=0; i<n; i++){
    a[i] = AB[i].a;
    b[i] = AB[i].b;
  }

  free(AB);
}



void qsort_double3(long n, double *a, int *b, int *c)
{
  int i;
  dlist3 *ABC;

  ABC = (dlist3*)malloc(sizeof(dlist3)*n);

  for (i=0; i<n; i++){
    ABC[i].a = a[i+1];
    ABC[i].b = b[i+1];
    ABC[i].c = c[i+1];
  }

  qsort(ABC, n, sizeof(dlist3), (int(*)(const void*, const void*))dlists_cmp);

  for (i=0; i<n; i++){
    a[i+1] = ABC[i].a;
    b[i+1] = ABC[i].b;
    c[i+1] = ABC[i].c;
  }

  free(ABC);
}



void qsort_double3B(long n, double *a, int *b, int *c)
{
  int i;
  dlist3 *ABC;

  ABC = (dlist3*)malloc(sizeof(dlist3)*n);

  for (i=0; i<n; i++){
    ABC[i].a = a[i];
    ABC[i].b = b[i];
    ABC[i].c = c[i];
  }

  qsort(ABC, n, sizeof(dlist3), (int(*)(const void*, const void*))dlists_cmp);

  for (i=0; i<n; i++){
    a[i] = ABC[i].a;
    b[i] = ABC[i].b;
    c[i] = ABC[i].c;
  }

  free(ABC);
}

int dlists_cmp(const dlists *x, const dlists *y)
{
  return (x->a < y->a ? -1 :
          y->a < x->a ?  1 : 0);
}

int dlists_cmp2(const dlists *x, const dlists *y)
{
  return (x->a < y->a ?  1 :
          y->a < x->a ? -1 : 0);
}


int ilists_cmp(const ilists *x, const ilists *y)
{
  return (x->a < y->a ? -1 :
          y->a < x->a ?  1 : 0);
}
