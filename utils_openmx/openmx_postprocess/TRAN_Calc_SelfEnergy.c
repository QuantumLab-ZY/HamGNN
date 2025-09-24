/**********************************************************************
  TRAN_Calc_SelfEnergy.c:

  TRAN_Calc_SelfEnergy.c is a subroutine to calculate selfenergy of the C region. 

   e.g., sigma_L = (w SCL-HCL) G_L(w) ( w SLC-HLC ) 

  Log of TRAN_Calc_SelfEnergy.o

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "lapack_prototypes.h"


void TRAN_Calc_SelfEnergy( 
			  /* input */
			  dcomplex w,      /* freq, imag>0 */
			  int ne,     
			  dcomplex *gr,    /* green function of electrode, size=ne*ne */
			  int nc,
			  dcomplex *hce,   /* e.g., HCL, size=nc*ne */
			  dcomplex *sce,   /* e.g., SCL, size=nc*ne */
			  /* output */
			  dcomplex *sigma  /* e.g., Sigma_L , size=nc*nc */
			  )

#define hce_ref(i,j) hce[ nc*((j)-1)+(i)-1]
#define sce_ref(i,j) sce[ nc*((j)-1)+(i)-1]

#define gr_ref(i,j) gr[ ne*((j)-1) +(i)-1 ]
#define sigma_ref(i,j)  sigma[ nc*((j)-1) + (i)-1 ]

#define wSH1_ref(i,j) wSH1[ nc*((j)-1) + (i)-1 ]
#define wSH2_ref(i,j) wSH2[ ne*((j)-1) + (i)-1 ]

{
  int i,j;
  dcomplex a,b;
  dcomplex *wSH1;
  dcomplex *wSH2;
  dcomplex *wSHG;

  a.r=1.0; a.i=0.0; 
  b.r=0.0; b.i=0.0;

  /* calculate (w SCL-HCL) G_L ( w SLC-HLC ) */

  wSH1 = (dcomplex*)malloc(sizeof(dcomplex)*nc*ne);
  wSH2 = (dcomplex*)malloc(sizeof(dcomplex)*ne*nc);
  wSHG = (dcomplex*)malloc(sizeof(dcomplex)*nc*ne);

  /* wSH1 = (w SCL-HCL) */

  for (i=1;i<=nc;i++) {
    for (j=1;j<=ne;j++) {
      wSH1_ref(i,j).r = w.r*sce_ref(i,j).r - w.i*sce_ref(i,j).i - hce_ref(i,j).r;
      wSH1_ref(i,j).i = w.i*sce_ref(i,j).r + w.r*sce_ref(i,j).i - hce_ref(i,j).i;
    }
  }

  /* wSH2 = ( w SLC-HLC ) */

  for (i=1;i<=ne;i++) {
    for (j=1;j<=nc;j++) {
      /*  wSH2  ne*nc
       *  sce   nc*ne 
       */
      /* taking accout of complex conjugate of S and H */
      wSH2_ref(i,j).r = w.r*sce_ref(j,i).r + w.i*sce_ref(j,i).i - hce_ref(j,i).r;
      wSH2_ref(i,j).i = w.i*sce_ref(j,i).r - w.r*sce_ref(j,i).i + hce_ref(j,i).i;
    }
  }

  /*  (w SCL-HCL) G_L  */
  /* 
   * wSH1  nc x ne
   * gr    ne x ne
   * wSHG  nc x ne 
   */

  F77_NAME(zgemm,ZGEMM)("N","N",&nc,&ne,&ne,&a,wSH1,&nc, gr, &ne, &b, wSHG, &nc);

  /*  (w SCL-HCL) G_L  * ( w SLC-HLC ) */
  /*
   *  wSHG  nc x ne
   *  wSH2  ne x nc
   * sigma  nc x nc
   */

  F77_NAME(zgemm,ZGEMM)("N","N",&nc,&nc,&ne,&a,wSHG, &nc, wSH2, &ne, &b, sigma, &nc);

  free(wSHG);
  free(wSH2);
  free(wSH1);
}

