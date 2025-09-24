#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "lapack_prototypes.h"

#define PI            3.1415926535897932384626

void TRAN_Calc_Sinv(
  int NUM_c,
  dcomplex *SCC,
  dcomplex *Sinv)
{
  INTEGER LWORK;
  dcomplex *WORK;
  double *RWORK;
  INTEGER INFO;
  double *eval;
  dcomplex *matrix1;

  int i_one;
  dcomplex one, zero;
  int i, j;
  int NUM_c2;

  eval = (double*)malloc(sizeof(double)*NUM_c);
  matrix1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);

  i_one = 1;
  NUM_c2 = NUM_c * NUM_c;
  one.r = 1.0;  one.i = 0.0;
  zero.r = 0.0;  zero.i = 0.0;

  F77_NAME(zcopy, ZCOPY)(&NUM_c2, SCC, &i_one, matrix1, &i_one);

  LWORK = 3 * NUM_c;
  WORK = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*(3 * NUM_c - 2));
  F77_NAME(zheev, ZHEEV)("V", "U", &NUM_c, matrix1, &NUM_c, eval, WORK, &LWORK, RWORK, &INFO);

  for (i = 0; i < NUM_c; i++){
    if (eval[i] < 0.000001) {
      eval[i] = 0.0;
    } /* if (eval[i] < 0.000001) */
    else{
      eval[i] = 1.0 / sqrt(eval[i]);
    } /* else if (eval[i] > 0.000001) */
    for (j = 0; j < NUM_c; j++){
      matrix1[j + NUM_c * i].r = eval[i] * matrix1[j + NUM_c * i].r;
      matrix1[j + NUM_c * i].i = eval[i] * matrix1[j + NUM_c * i].i;
    }
  } /* for (i = 0; i < NUM_c; i++) */

  F77_NAME(zgemm, ZGEMM)("N", "C", &NUM_c, &NUM_c, &NUM_c, &one, matrix1, &NUM_c,
    matrix1, &NUM_c, &zero, Sinv, &NUM_c);

  free(matrix1);
  free(eval);
  free(WORK);
  free(RWORK);
} /* int TRAN_Calc_Sinv */

static void TRAN_MatProds3(int NUM_c, char *trans,
  dcomplex *In1, dcomplex *In2, dcomplex *In3,
  dcomplex *Buf, dcomplex *Out1)
{
  dcomplex one, zero;

  one.r = 1.0;  one.i = 0.0;
  zero.r = 0.0;  zero.i = 0.0;
  
  F77_NAME(zgemm, ZGEMM)(trans, trans, &NUM_c, &NUM_c, &NUM_c, &one, In2, &NUM_c,
    In3, &NUM_c, &zero, Buf, &NUM_c);
  F77_NAME(zgemm, ZGEMM)(trans, "N", &NUM_c, &NUM_c, &NUM_c, &one, In1, &NUM_c,
    Buf, &NUM_c, &zero, Out1, &NUM_c);
}

static void TRAN_MatProds5(int NUM_c, char *trans,
  dcomplex *In1, dcomplex *In2, dcomplex *In3, dcomplex *In4, dcomplex *In5,
  dcomplex *Buf, dcomplex *Out1)
{
  dcomplex one, zero;

  one.r = 1.0;  one.i = 0.0;
  zero.r = 0.0;  zero.i = 0.0;

  F77_NAME(zgemm, ZGEMM)(trans, trans, &NUM_c, &NUM_c, &NUM_c, &one, In4, &NUM_c,
    In5, &NUM_c, &zero, Buf, &NUM_c);
  F77_NAME(zgemm, ZGEMM)(trans, "N", &NUM_c, &NUM_c, &NUM_c, &one, In3, &NUM_c,
    Buf, &NUM_c, &zero, Out1, &NUM_c);
  F77_NAME(zgemm, ZGEMM)(trans, "N", &NUM_c, &NUM_c, &NUM_c, &one, In2, &NUM_c,
    Out1, &NUM_c, &zero, Buf, &NUM_c);
  F77_NAME(zgemm, ZGEMM)(trans, "N", &NUM_c, &NUM_c, &NUM_c, &one, In1, &NUM_c,
    Buf, &NUM_c, &zero, Out1, &NUM_c);
}

static void TRAN_Store_Matrix(int NUM_c, double fL, double fR, double Tran_current_energy_step,
  dcomplex *In, double **Out)
{
  int i, j;

  for (i = 0; i < NUM_c; i++) {
    for (j = 0; j < NUM_c; j++) {
      Out[j][i] += In[j + NUM_c*i].r * (fR - fL) * Tran_current_energy_step / (2.0 * PI);
    }
  }
}

void TRAN_Calc_CurrentDensity(
  int NUM_c, 
  dcomplex *GC, 
  dcomplex *GammaL, 
  dcomplex *GammaR, 
  dcomplex *VCC, 
  dcomplex *Sinv, 
  double *kvec,
  double fL,
  double fR,
  double Tran_current_energy_step,
  double ***JLocSym, 
  double **JLocAsym, 
  double **RhoNL, 
  double ***Jmat
  )
{
  int i, j, iaxis, minusk;
  dcomplex *matrix1, *matrix2, *matrix3, *GCdag;

  matrix1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  matrix2 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  matrix3 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCdag = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  for (i = 0; i < NUM_c*NUM_c; i++){
    matrix1[i].r = 0.0;    matrix1[i].i = 0.0;
    matrix2[i].r = 0.0;    matrix2[i].i = 0.0;
    matrix3[i].r = 0.0;    matrix3[i].i = 0.0;
  }
  for (i = 0; i < NUM_c; i++) {
    for (j = 0; j < NUM_c; j++) {
      GCdag[i + NUM_c*j].r = GC[j + NUM_c*i].r;
      GCdag[i + NUM_c*j].i = -GC[j + NUM_c*i].i;
    }
  }

  if (fabs(kvec[0]) > 0.000001 && fabs(kvec[1]) > 0.000001) minusk = 1;
  else minusk = 0;

  /*
   kvec[] * (GC * GammaR * GC^+ - GC^T * (GC^+ * GammaR)^T)
  */
  TRAN_MatProds3(NUM_c, "N", GC, GammaR, GCdag, matrix1, matrix2);
  if (minusk == 1) TRAN_MatProds3(NUM_c, "T", GC, GammaR, GCdag, matrix1, matrix3);
  for (iaxis = 0; iaxis < 2; iaxis++) {
    for (i = 0; i < NUM_c*NUM_c; i++) 
      matrix1[i].r = 2.0 * kvec[iaxis] * (matrix2[i].r - matrix3[i].r);
    TRAN_Store_Matrix(NUM_c, fL, fR, Tran_current_energy_step, matrix1, JLocSym[iaxis]);
  }
  /*
  GC * GammaR * GC^+ + GC^T * GammaR^T * GC^+^T
  */
  for (i = 0; i < NUM_c*NUM_c; i++)
    matrix1[i].r = matrix2[i].i + matrix3[i].i;
  TRAN_Store_Matrix(NUM_c, fL, fR, Tran_current_energy_step, matrix1, JLocAsym);
  /*
  Sinv * VCC * GC * GammaR * GC^+ + Sinv^T * VCC^T * GC^T * GammaR^T * GC^+^T
  */
  TRAN_MatProds5(NUM_c, "N", Sinv, VCC, GC, GammaR, GCdag, matrix1, matrix2);
  if (minusk == 1) TRAN_MatProds5(NUM_c, "T", Sinv, VCC, GC, GammaR, GCdag, matrix1, matrix3);
  for (i = 0; i < NUM_c*NUM_c; i++)
    matrix1[i].r = -2.0 * (matrix2[i].i + matrix3[i].i);
  TRAN_Store_Matrix(NUM_c, fL, fR, Tran_current_energy_step, matrix1, RhoNL);
  /*
   GC * GammaR * GC^+ * GammaL * Sinv + GC^T * GammaR^T * GC^+^T * GammaL^T * Sinv^T
  */
  TRAN_MatProds5(NUM_c, "N", GC, GammaR, GCdag, GammaL, Sinv, matrix1, matrix2);
  if (minusk == 1) TRAN_MatProds5(NUM_c, "T", GC, GammaR, GCdag, GammaL, Sinv, matrix1, matrix3);
  for (i = 0; i < NUM_c*NUM_c; i++)
    matrix1[i].r = matrix2[i].r + matrix3[i].r;
  TRAN_Store_Matrix(NUM_c, fL, fR, Tran_current_energy_step, matrix1, Jmat[0]);
  /*
  GC * GammaL * GC^+ * GammaR * Sinv + GC^T * GammaL^T * GC^+^T *GammaR^T * Sinv^T
  */
  TRAN_MatProds5(NUM_c, "N", GC, GammaL, GCdag, GammaR, Sinv, matrix1, matrix2);
  if (minusk == 1) TRAN_MatProds5(NUM_c, "T", GC, GammaL, GCdag, GammaR, Sinv, matrix1, matrix3);
  for (i = 0; i < NUM_c*NUM_c; i++)
    matrix1[i].r = matrix2[i].r + matrix3[i].r;
  TRAN_Store_Matrix(NUM_c, fL, fR, Tran_current_energy_step, matrix1, Jmat[1]);

  free(matrix1);
  free(matrix2);
  free(matrix3);
  free(GCdag);
}

static void TRAN_Store_Matrix_NC(int NUM_cs, double fL, double fR, double Tran_current_energy_step,
  dcomplex *In, double ***Out)
{
  int i, j, ispin;

  for (ispin = 0; ispin < 2; ispin++) {
    for (i = 0; i < NUM_cs; i++) {
      for (j = 0; j < NUM_cs; j++) {
        Out[ispin][j][i] += In[(NUM_cs * ispin + j) + NUM_cs*2*(NUM_cs * ispin + i)].r
          * (fR - fL) * Tran_current_energy_step / (2.0 * PI);
      }/*for (j = 0; j < NUM_cs; j++)*/
    }/*for (i = 0; i < NUM_cs; i++)*/
  }/*for (ispin = 0; ispin < 2; ispin++)*/

  for (i = 0; i < NUM_cs; i++) {
    for (j = 0; j < NUM_cs; j++) {

      Out[2][j][i] += In[(NUM_cs + j) + NUM_cs*2 *i].r
        * (fR - fL) * Tran_current_energy_step / (2.0 * PI);

      Out[3][j][i] += In[(NUM_cs + j) + NUM_cs*2*i].i
        * (fR - fL) * Tran_current_energy_step / (2.0 * PI);

    }/*for (j = 0; j < NUM_cs; j++)*/
  }/*for (i = 0; i < NUM_cs; i++)*/
}

static void TRAN_Store_Matrix_NC_vec(int NUM_cs, double fL, double fR, double Tran_current_energy_step,
  dcomplex *In, double ****Out, int iaxis)
{
  int i, j, ispin;

  for (ispin = 0; ispin < 2; ispin++) {
    for (i = 0; i < NUM_cs; i++) {
      for (j = 0; j < NUM_cs; j++) {
        Out[ispin][iaxis][j][i] += In[(NUM_cs * ispin + j) + NUM_cs * 2 * (NUM_cs * ispin + i)].r
          * (fR - fL) * Tran_current_energy_step / (2.0 * PI);
      }/*for (j = 0; j < NUM_cs; j++)*/
    }/*for (i = 0; i < NUM_cs; i++)*/
  }/*for (ispin = 0; ispin < 2; ispin++)*/

  for (i = 0; i < NUM_cs; i++) {
    for (j = 0; j < NUM_cs; j++) {

      Out[2][iaxis][j][i] += In[(NUM_cs + j) + NUM_cs * 2 * i].r
        * (fR - fL) * Tran_current_energy_step / (2.0 * PI);

      Out[3][iaxis][j][i] += In[(NUM_cs + j) + NUM_cs * 2 * i].i
        * (fR - fL) * Tran_current_energy_step / (2.0 * PI);

    }/*for (j = 0; j < NUM_cs; j++)*/
  }/*for (i = 0; i < NUM_cs; i++)*/
}

void TRAN_Calc_CurrentDensity_NC(
  int NUM_c,
  dcomplex *GC,
  dcomplex *GammaL,
  dcomplex *GammaR,
  dcomplex *VCC,
  dcomplex *Sinv,
  double *kvec,
  double fL,
  double fR,
  double Tran_current_energy_step,
  double ****JLocSym,
  double ***JLocAsym,
  double ***RhoNL,
  double ****Jmat
  )
{
  int i, j, iaxis, NUM_cs;
  dcomplex *matrix1, *matrix2, *GCdag;

  NUM_cs = NUM_c / 2;

  matrix1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  matrix2 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCdag = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  for (i = 0; i < NUM_c*NUM_c; i++) {
    matrix1[i].r = 0.0;    matrix1[i].i = 0.0;
    matrix2[i].r = 0.0;    matrix2[i].i = 0.0;
  }
  for (i = 0; i < NUM_c; i++) {
    for (j = 0; j < NUM_c; j++) {
      GCdag[i + NUM_c*j].r = GC[j + NUM_c*i].r;
      GCdag[i + NUM_c*j].i = -GC[j + NUM_c*i].i;
    }
  }
  /*
  kvec[] * (GC * GammaR * GC^+ - GC^T * GammaR^T * GC^+^T)
  */
  TRAN_MatProds3(NUM_c, "N", GC, GammaR, GCdag, matrix1, matrix2);
  for (iaxis = 0; iaxis < 2; iaxis++) {
    for (i = 0; i < NUM_c*NUM_c; i++) {
      matrix1[i].r = 2.0 * kvec[iaxis] * matrix2[i].r;
      matrix1[i].i = 2.0 * kvec[iaxis] * matrix2[i].i;
    }
    TRAN_Store_Matrix_NC_vec(NUM_cs, fL, fR, Tran_current_energy_step, matrix1, JLocSym, iaxis);
  }
  /*
  GC * GammaR * GC^+ + GC^T * GammaR^T * GC^+^T
  */
  for (i = 0; i < NUM_c*NUM_c; i++) {
    matrix1[i].r = matrix2[i].i;
    matrix1[i].i = - matrix2[i].r;
  }
  TRAN_Store_Matrix_NC(NUM_cs, fL, fR, Tran_current_energy_step, matrix1, JLocAsym);
  /*
  Sinv * VCC * GC * GammaR * GC^+ + Sinv^T * VCC^T * GC^T * GammaR^T * GC^+^T
  */
  TRAN_MatProds5(NUM_c, "N", Sinv, VCC, GC, GammaR, GCdag, matrix1, matrix2);
  for (i = 0; i < NUM_c*NUM_c; i++) {
    matrix1[i].r = -2.0 * matrix2[i].i;
    matrix1[i].i = 2.0 * matrix2[i].r;
  }
  TRAN_Store_Matrix_NC(NUM_cs, fL, fR, Tran_current_energy_step, matrix1, RhoNL);
  /*
  GC * GammaR * GC^+ * GammaL * Sinv + GC^T * GammaR^T * GC^+)^T * GammaL^T * Sinv^T
  */
  TRAN_MatProds5(NUM_c, "N", GC, GammaR, GCdag, GammaL, Sinv, matrix1, matrix2);
  for (i = 0; i < NUM_c*NUM_c; i++) {
    matrix1[i].r = matrix2[i].r;
    matrix1[i].i = matrix2[i].i;
  }
  TRAN_Store_Matrix_NC_vec(NUM_cs, fL, fR, Tran_current_energy_step, matrix1, Jmat, 0);
  /*
  GC * GammaL * GC^+ * GammaR * Sinv + GC^T * GammaL^T * GC^+^T * GammaR^T * Sinv^T
  */
  TRAN_MatProds5(NUM_c, "N", GC, GammaL, GCdag, GammaR, Sinv, matrix1, matrix2);
  for (i = 0; i < NUM_c*NUM_c; i++) {
    matrix1[i].r = matrix2[i].r;
    matrix1[i].i = matrix2[i].i;
  }
  TRAN_Store_Matrix_NC_vec(NUM_cs, fL, fR, Tran_current_energy_step, matrix1, Jmat, 1);

  free(matrix1);
  free(matrix2);
  free(GCdag);
}/*TRAN_Calc_CurrentDensity_NC*/
