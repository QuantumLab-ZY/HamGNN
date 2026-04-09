/**********************************************************************
  TRAN_Set_Value.c:

  TRAN_Set_Value.c is a subroutine to initialize A.

  Log of TRAN_Set_Value.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <mpi.h>
#include "tran_prototypes.h"

/*
 * used to initialize A
 */
void TRAN_Set_Value_double(dcomplex *A, int n, double a, double b)
{
  int i;
  for(i=0; i<n; i++) {
    A[i].r = a;
    A[i].i = b;
  }
}
