/**********************************************************************
  TRAN_Calc_CentGreen.c:

  TRAN_Calc_CentGreen.c is a subroutine to calculate the Green's 
  function of the central part: G(w) = ( w SCC - HCC - SigmaL -SigmaR ) ^-1 

  Log of TRAN_Calc_CentGreen.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "tran_prototypes.h"


void TRAN_Calc_CentGreen(
			 /* input */
			 dcomplex w,
			 int nc, 
			 dcomplex *sigmaL,
			 dcomplex *sigmaR, 
			 dcomplex *HCC,
			 dcomplex *SCC,

                         /* output */
			 dcomplex *GC)

#define HCC_ref(i,j)    HCC[nc*((j)-1)+(i)-1]
#define SCC_ref(i,j)    SCC[nc*((j)-1)+(i)-1]
#define GC_ref(i,j)     GC[nc*((j)-1)+(i)-1]
#define sigmaL_ref(i,j) sigmaL[nc*((j)-1)+(i)-1]
#define sigmaR_ref(i,j) sigmaR[nc*((j)-1)+(i)-1]

{
  int i,j;

  /* w SCC - HCC - SigmaL - SigmaR */
    
  for (i=1; i<=nc; i++) {
    for (j=1; j<=nc; j++) {

      GC_ref(i,j).r = w.r*SCC_ref(i,j).r - w.i*SCC_ref(i,j).i - HCC_ref(i,j).r
                     - sigmaL_ref(i,j).r - sigmaR_ref(i,j).r;
      GC_ref(i,j).i = w.r*SCC_ref(i,j).i + w.i*SCC_ref(i,j).r - HCC_ref(i,j).i
                     - sigmaL_ref(i,j).i - sigmaR_ref(i,j).i;
    }
  }

  Lapack_LU_Zinverse(nc,GC);

}



