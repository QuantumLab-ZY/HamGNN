/**********************************************************************
  TRAN_Calc_OneTransmission.c:

  TRAN_Calc_OneTransmission.c is a subroutine to calculate the transmission.

   input: SigmaL, SigmaR, G_CC^R(w)
   assuming w= w+i delta 

   Gamma(w) = i (Sigma^R(w) - Sigma^A(w))
 
   T(w) = Trace[ Gamma_L(w) G^R(w) Gamma_R(w) G^A(w) ]

   work: v1,v2  dcomplex size=nc*nc


  Log of TRAN_Calc_OneTransmission.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "lapack_prototypes.h"

/* 
 * calculate transmission 
 *
 * input: SigmaL, SigmaR, G_CC^R(w)
 *   assuming w= w+i delta 
 *
 *   Gamma(w) = i (Sigma^R(w) - Sigma^A(w))
 * 
 *   T(w) = Trace[ Gamma_L(w) G^R(w) Gamma_R(w) G^A(w) ]
 *
 * work: v1,v2  dcomplex size=nc*nc
 */

void TRAN_Calc_OneTransmission(int nc, 
			       dcomplex *SigmaL_R,   /* at w, changed when exit */
			       dcomplex *SigmaL_A,   /* at w, changed when exit */
			       dcomplex *SigmaR_R,   /* at w, changed when exit */
			       dcomplex *SigmaR_A,   /* at w, changed when exit */
			       dcomplex *GC_R,       /* at w, changed when exit */
			       dcomplex *GC_A,       /* at w, changed when exit */
			       dcomplex *v1,         /* work */
			       dcomplex *v2,         /* work */
			       dcomplex *value       /* output, transmission */
			       )

#define SigmaL_R_ref(i,j) SigmaL_R[nc*((j)-1)+(i)-1]
#define SigmaL_A_ref(i,j) SigmaL_A[nc*((j)-1)+(i)-1]
#define SigmaR_R_ref(i,j) SigmaR_R[nc*((j)-1)+(i)-1]
#define SigmaR_A_ref(i,j) SigmaR_A[nc*((j)-1)+(i)-1]
#define GC_R_ref(i,j)     GC_R[nc*((j)-1)+(i)-1]
#define GC_A_ref(i,j)     GC_A[nc*((j)-1)+(i)-1]
#define v1_ref(i,j)       v1[nc*((j)-1)+(i)-1]

{
  int i,j;
  dcomplex alpha,beta;
  double tmpr,tmpi;

  alpha.r = 1.0;  alpha.i = 0.0;
  beta.r  = 0.0;  beta.i  = 0.0;

  /* Gamma_l = i (Sigma_l^R-Sigma_l^A) */
   
  for (j=1; j<=nc; j++) {
    for (i=1; i<=nc; i++) {

      tmpr = -(SigmaL_R_ref(i,j).i - SigmaL_A_ref(i,j).i);
      tmpi =  (SigmaL_R_ref(i,j).r - SigmaL_A_ref(i,j).r);

      SigmaL_R_ref(i,j).r = tmpr;
      SigmaL_R_ref(i,j).i = tmpi;

      tmpr = -(SigmaR_R_ref(i,j).i - SigmaR_A_ref(i,j).i);
      tmpi =  (SigmaR_R_ref(i,j).r - SigmaR_A_ref(i,j).r);

      SigmaR_R_ref(i,j).r = tmpr;
      SigmaR_R_ref(i,j).i = tmpi;
    }
  }

  /* sigma -> gamma */

  /* transmission= tr[GammaL G^R GammaR G^A] */

  /* GammaL * GR */
  F77_NAME(zgemm,ZGEMM)("N","N", &nc,&nc,&nc, &alpha, SigmaL_R, &nc, GC_R, &nc, &beta, v1, &nc );


  /* (GammaL * GR ) * GammaR */
  F77_NAME(zgemm,ZGEMM)("N","N", &nc,&nc,&nc, &alpha, v1, &nc, SigmaR_R, &nc, &beta, v2, &nc );

  /* (GammaL G^R GammaR ) * G^A */

  F77_NAME(zgemm,ZGEMM)("N","N", &nc,&nc,&nc, &alpha, v2, &nc, GC_A, &nc, &beta, v1, &nc );

  value->r=0.0; value->i = 0.0;

  /* trace */
  for (i=1;i<=nc;i++) {
    value->r += v1_ref(i,i).r;
    value->i += v1_ref(i,i).i;
  } 

}


