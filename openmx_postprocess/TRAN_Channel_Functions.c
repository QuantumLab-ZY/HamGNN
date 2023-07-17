/**********************************************************************
TRAN_Channel_Functions.c:

 Routines used in MTRAN_EigenChannel.

Log of TRAN_Channel_Functions.c:
 
  xx/Xxx/2015  Released by M. Kawamura

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "lapack_prototypes.h"


/* Diagonalize ALbar -> its eigenvector */
/* Then scale it : Albar_{ml} -> ALbar * sqrt(eig_l)  */

void TRAN_Calc_Diagonalize(
  int n, 
  dcomplex *evec, 
  double *eval, 
  int lscale)
#define evec_ref(i,j) evec[n * (j) + (i)]
{
  INTEGER LWORK;
  dcomplex *WORK;
  double *RWORK;
  INTEGER INFO;
  int i, j;
  double sortmp;

  LWORK = 3 * n;
  WORK = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*(3 * n - 2));
  F77_NAME(zheev, ZHEEV)("V", "U", &n, evec, &n, eval, WORK, &LWORK, RWORK, &INFO);

  /* Re-order eigenvalues & vectors as decreasing order */
  i = 1;
  for (j = 0; j < n / 2; j++){
    sortmp = eval[j];
    eval[j] = eval[n - 1 - j];
    eval[n - 1 - j] = sortmp;
    F77_NAME(zcopy, ZCOPY)(&n, &evec_ref(0, j), &i, WORK, &i);
    F77_NAME(zcopy, ZCOPY)(&n, &evec_ref(0, n - 1 - j), &i, &evec_ref(0, j), &i);
    F77_NAME(zcopy, ZCOPY)(&n, WORK, &i, &evec_ref(0, n - 1 - j), &i);
  } /* for (j = 0; j < n / 2; j++) */

  if (INFO != 0){
    printf("Zheev error in TRAN_Calc_Diagonalization. info = %d, \n", INFO);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(0);
  }

  if (lscale == 1){
    for (j = 0; j < n; j++){
      for (i = 0; i < n; i++){
        evec_ref(i, j).r = evec_ref(i, j).r * sqrt(fabs(eval[j]));
        evec_ref(i, j).i = evec_ref(i, j).i * sqrt(fabs(eval[j]));
      } /* for (i = 0, i < n; i++) */
    } /* for (j = 0; j < n; j++) */
  } /* if (lscale == 1) */

  free(WORK);
  free(RWORK);

} /* TRAN_Calc_Diagonalize */

/* Define thedimention of ortogonal basis space */

int TRAN_Calc_OrtSpace(
  int NUM_c, 
  dcomplex *SCC, 
  dcomplex *rtS, 
  dcomplex *rtSinv)
#define  Sevec_ref(i,j)  Sevec[NUM_c * (j) + (i)]
#define    rtS_ref(i,j)    rtS[NUM_c * (j) + (i)]
#define rtSinv_ref(i,j) rtSinv[NUM_c * (j) + (i)]
{
  int NUM_cs;

  double *eval;
  dcomplex *Sevec;

  int one;
  int i, j;
  int NUM_c2;

  eval = (double*)malloc(sizeof(double)*NUM_c);
  Sevec = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  
  one = 1;
  NUM_c2 = NUM_c * NUM_c;

  F77_NAME(zcopy, ZCOPY)(&NUM_c2, SCC, &one, Sevec, &one);

  TRAN_Calc_Diagonalize(NUM_c, Sevec, eval, 0);

  NUM_cs = NUM_c;
  for (i = 0; i < NUM_c; i++){
    if (eval[i] < 0.000001) {
      NUM_cs = i;
      break;
    } /* if (eval[i] < 0.000001) */
    else{
      eval[i] = sqrt(eval[i]);
    } /* else if (eval[i] > 0.000001) */
  } /* for (i = 0; i < NUM_c; i++) */

  for (j = 0; j < NUM_cs; j++){
    for (i = 0; i < NUM_c; i++){
      rtS_ref(i, j).r = Sevec_ref(i, j).r * eval[j];
      rtS_ref(i, j).i = Sevec_ref(i, j).i * eval[j];
      rtSinv_ref(i, j).r = Sevec_ref(i, j).r / eval[j];
      rtSinv_ref(i, j).i = Sevec_ref(i, j).i / eval[j];
    } /* for (i = 0; i < NUM_c; i++) */
  } /* for (j = 0; j < NUM_cs; j++) */

  free(Sevec);
  free(eval);

  return NUM_cs;
} /* int TRAN_Calc_OrtSpace */

/* i(Sigma  Sigma^+) -> Sigma*/

void TRAN_Calc_Linewidth(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *SigmaR_R)
#define SigmaL_R_ref(i,j) SigmaL_R[NUM_c * (j) + (i)]
#define SigmaR_R_ref(i,j) SigmaR_R[NUM_c * (j) + (i)]
{
  int i, j;
  double gammar, gammai;

  for (j = 0; j < NUM_c; j++){
    for (i = 0; i < j; i++){
      gammar = - SigmaL_R_ref(i, j).i - SigmaL_R_ref(j, i).i;
      gammai =   SigmaL_R_ref(i, j).r - SigmaL_R_ref(j, i).r;
      SigmaL_R_ref(i, j).r =   gammar;
      SigmaL_R_ref(i, j).i =   gammai;
      SigmaL_R_ref(j, i).r =   gammar;
      SigmaL_R_ref(j, i).i = - gammai;

      gammar = -SigmaR_R_ref(i, j).i - SigmaR_R_ref(j, i).i;
      gammai = SigmaR_R_ref(i, j).r - SigmaR_R_ref(j, i).r;
      SigmaR_R_ref(i, j).r = gammar;
      SigmaR_R_ref(i, j).i = gammai;
      SigmaR_R_ref(j, i).r = gammar;
      SigmaR_R_ref(j, i).i = -gammai;
    } /* for (i = 0; i < j; i++) */
    SigmaL_R_ref(j, j).r = -2.0 * SigmaL_R_ref(j, j).i;
    SigmaL_R_ref(j, j).i = 0.0;

    SigmaR_R_ref(j, j).r = -2.0 * SigmaR_R_ref(j, j).i;
    SigmaR_R_ref(j, j).i = 0.0;
  } /* for (j = 0; j < NUM_c; j++) */

} /* TRAN_Calc_Linewidth */

/* G Gamma_L G^+ -> Sigma_L */

void TRAN_Calc_MatTrans(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *GC_R, 
  char* trans1, 
  char* trans2)
{
  dcomplex alpha, beta;
  dcomplex *v1;

  alpha.r = 1.0;  alpha.i = 0.0;
  beta.r = 0.0;  beta.i = 0.0;

  v1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  F77_NAME(zgemm, ZGEMM)(trans1, "N", &NUM_c, &NUM_c, &NUM_c, &alpha, GC_R, &NUM_c, 
    SigmaL_R, &NUM_c, &beta, v1, &NUM_c);

  F77_NAME(zgemm, ZGEMM)("N", trans2, &NUM_c, &NUM_c, &NUM_c, &alpha, v1, &NUM_c,
    GC_R, &NUM_c, &beta, SigmaL_R, &NUM_c);

  free(v1);

} /* void TRAN_Calc_MatTran */

/* L\"owdin orthogonalization */

void TRAN_Calc_LowdinOrt(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *SigmaR_R,
  int NUM_cs, 
  dcomplex *rtS, 
  dcomplex *rtSinv, 
  dcomplex *ALbar, 
  dcomplex *GamRbar)
{
  dcomplex alpha, beta;
  dcomplex *v1;

  alpha.r = 1.0;  alpha.i = 0.0;
  beta.r = 0.0;  beta.i = 0.0;

  v1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_c);

  /* Transform A_L into orthogonal basis space with S^{1/2}*/

  F77_NAME(zgemm, ZGEMM)("C", "N", &NUM_cs, &NUM_c, &NUM_c, &alpha, rtS, &NUM_c,
    SigmaL_R, &NUM_c, &beta, v1, &NUM_cs);
  F77_NAME(zgemm, ZGEMM)("N", "N", &NUM_cs, &NUM_cs, &NUM_c, &alpha, v1, &NUM_cs,
    rtS, &NUM_c, &beta, ALbar, &NUM_cs);

  /* Transform Gamma_R int orthogonal basis space wit S^{-1/2} */

  F77_NAME(zgemm, ZGEMM)("C", "N", &NUM_cs, &NUM_c, &NUM_c, &alpha, rtSinv, &NUM_c,
    SigmaR_R, &NUM_c, &beta, v1, &NUM_cs);
  F77_NAME(zgemm, ZGEMM)("N", "N", &NUM_cs, &NUM_cs, &NUM_c, &alpha, v1, &NUM_cs,
    rtSinv, &NUM_c, &beta, GamRbar, &NUM_cs);

  free(v1);
} /* void TRAN_Calc_LowdinOrt */

/* Transform eigenchannel into non-orthogonal basis space */

void TRAN_Calc_ChannelLCAO(
  int NUM_c, 
  int NUM_cs, 
  dcomplex *rtSinv, 
  dcomplex *ALbar,
  dcomplex *GamRbar, 
  double *eval, 
  dcomplex *GC_R, 
  int TRAN_Channel_Num, 
  dcomplex **EChannel, 
  double *eigentrans)
#define GC_R_ref(i,j) GC_R[NUM_c * (j) + (i)]
{
  int i, j;
  dcomplex alpha, beta;
  dcomplex *v1;
  double ecabs;
  dcomplex ecphase, ec0;

  alpha.r = 1.0;  alpha.i = 0.0;
  beta.r = 0.0;  beta.i = 0.0;

  v1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_cs);

  F77_NAME(zgemm, ZGEMM)("N", "N", &NUM_cs, &NUM_cs, &NUM_cs, &alpha, ALbar, &NUM_cs,
    GamRbar, &NUM_cs, &beta, v1, &NUM_cs);

  F77_NAME(zgemm, ZGEMM)("N", "N", &NUM_c, &NUM_cs, &NUM_cs, &alpha, rtSinv, &NUM_c,
    v1, &NUM_cs, &beta, GC_R, &NUM_c);

  /* Adjust phase in each eigenchannel */
  for (j = 0; j < NUM_cs; j++){
    ecphase.r = 0.0;
    ecphase.i = 0.0;
    for (i = 0; i < NUM_c; i++){
      if (GC_R_ref(i, j).i > 0.0){
        ecphase.r += GC_R_ref(i, j).r;
        ecphase.i += GC_R_ref(i, j).i;
      }
      else{
        ecphase.r -= GC_R_ref(i, j).r;
        ecphase.i -= GC_R_ref(i, j).i;
      }
    }

    ecabs = sqrt(ecphase.r * ecphase.r + ecphase.i * ecphase.i);
    ecphase.r /= ecabs;
    ecphase.i /= ecabs;

    for (i = 0; i < NUM_c; i++){
      ec0.r = GC_R_ref(i, j).r;
      ec0.i = GC_R_ref(i, j).i;
      GC_R_ref(i, j).r = ec0.r * ecphase.r + ec0.i * ecphase.i;
      GC_R_ref(i, j).i = - ec0.r * ecphase.i + ec0.i * ecphase.r;
    }
  }

  /* Fill Trancated dimentions with 0 */
  for (j = NUM_cs; j < NUM_c; j++){
    eval[j] = 0.0;
    for (i = 0; i < NUM_c; i++){
      GC_R_ref(i, j).r = 0.0;
      GC_R_ref(i, j).i = 0.0;
    }
  }

    /* Store for cube output */

  for (j = 0; j < TRAN_Channel_Num; j++){
    eigentrans[j] = eval[j];
    for (i = 0; i < NUM_c; i++){
      EChannel[j][i].r = GC_R_ref(i, j).r;
      EChannel[j][i].i = GC_R_ref(i, j).i;
    }
  }

  free(v1);

} /* void TRAN_Calc_ChannelLCAO */
