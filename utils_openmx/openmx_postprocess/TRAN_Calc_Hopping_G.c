/**********************************************************************
  TRAN_Calc_Hopping_G.c:

  TRAN_Calc_Hopping_G.c is a subroutine to calculate off-diagonal Green's
  funtions, GCL and GCR, which are used to calculate the density of states
  near the boundary regions.  

  Log of TRAN_Calc_Hopping_G.c:

     24/July/2008  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "lapack_prototypes.h"


/* calculate -gc (w sce - hce) gs */

void TRAN_Calc_Hopping_G(
                         /* input */
			 dcomplex w,
			 int ne,          /* size of electrode */ 
			 int nc,          /* size of central region */
			 dcomplex *gs,    /* surface green function of electrode, size=ne*ne */
			 dcomplex *gc,    /* green function of central region, size=nc*nc    */
			 dcomplex *hce,   /* e.g., HCL,    size=nc*ne */
			 dcomplex *sce,   /* e.g., SCL,    size=nc*ne */

                         /* output */
			 dcomplex *gh     /* e.g., GCL_R , size=nc*ne */
			 )

#define hce_ref(i,j)    hce[ nc*((j)-1)+(i)-1]
#define sce_ref(i,j)    sce[ nc*((j)-1)+(i)-1]
#define wSH1_ref(i,j)   wSH1[ nc*((j)-1) + (i)-1 ]

{
  int i,j;
  dcomplex a,b;
  dcomplex *wSH1;
  dcomplex *wSHG;

  a.r = 1.0; a.i = 0.0; 
  b.r = 0.0; b.i = 0.0;

  /* allocation of arrays */

  wSH1 = (dcomplex*)malloc(sizeof(dcomplex)*nc*ne);
  wSHG = (dcomplex*)malloc(sizeof(dcomplex)*nc*ne);

  /* wSH1 = -(w sce - hce) */

  for (i=1; i<=nc; i++) {
    for (j=1; j<=ne; j++) {
      wSH1_ref(i,j).r = -(w.r*sce_ref(i,j).r - w.i*sce_ref(i,j).i - hce_ref(i,j).r);
      wSH1_ref(i,j).i = -(w.i*sce_ref(i,j).r + w.r*sce_ref(i,j).i - hce_ref(i,j).i);
    }
  }

  /* gc wSH1 -> wSHG */
  /*********************** 
    gc    nc x nc
    wSH1  nc x ne
    wSHG  nc x ne 
  ************************/

  F77_NAME(zgemm,ZGEMM)( "N","N", &nc, &ne, &nc, &a, gc, &nc, wSH1, &nc, &b, wSHG, &nc );

  /* wSHG * gs -> gh  */
  /************************
    wSHG  nc x ne
    gs    ne x ne
    gh    nc x ne
  ************************/

  F77_NAME(zgemm,ZGEMM)( "N","N", &nc, &ne, &ne, &a, wSHG, &nc, gs, &ne, &b, gh, &nc );

  /* freeing of arrays */

  free(wSHG);
  free(wSH1);
}

