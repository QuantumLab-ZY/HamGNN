#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h> 
#include <mpi.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "tran_prototypes.h"
#include "tran_variables.h"

void FFT2D_Poisson(double *ReRhor, double *ImRhor,
  double *ReRhok, double *ImRhok);

static void TRAN_Calc_Orb2Real(int NUM_c, int *MP, double **JOrb, double *JReal,
  dcomplex *SCC)
{
  int myid;
  int Mc_AN, Gc_AN;
  int h_AN, Gh_AN, Mh_AN, Rnh;
  int Nog, Nc, Nh;
  int i, j, GN, Rn;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Rnh = ncn[Gc_AN][h_AN];

      if (atv_ijk[Rnh][1] != 0) continue;

      for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++){

        Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
        Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
        GN = GridListAtom[Mc_AN][Nc];

        Rn = CellListAtom[Mc_AN][Nc];

        if (G2ID[Gh_AN] == myid){
          for (i = 0; i < Spe_Total_CNO[WhatSpecies[Gc_AN]]; i++){
            for (j = 0; j < Spe_Total_CNO[WhatSpecies[Gh_AN]]; j++){

              JReal[GN] += JOrb[MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                * Orbs_Grid[Mc_AN][Nc][i] * Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */

              /*if (atv_ijk[Rn][1] == 0) JReal[GN] += SumJ0;*/

            }/*for (j = 0; j < NO1; j++)*/
          }/*for (i = 0; i < NO0; i++)*/
        }/*if (G2ID[Gh_AN] == myid)*/
        else{
          for (i = 0; i < Spe_Total_CNO[WhatSpecies[Gc_AN]]; i++){
            for (j = 0; j < Spe_Total_CNO[WhatSpecies[Gh_AN]]; j++){

              JReal[GN] += JOrb[MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                * Orbs_Grid[Mc_AN][Nc][i] * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */

              /*if (atv_ijk[Rn][1] == 0) JReal[GN] += SumJ0;*/

            } /*for (j = 0; j < NO1; j++)*/
          } /*for (i = 0; i < NO0; i++)*/
        } /*if (G2ID[Gh_AN] != myid)*/

      } /*for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++)*/
    } /*for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)*/
  } /*for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)*/
 
  MPI_Allreduce(MPI_IN_PLACE, JReal, TNumGrid, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

} /*static void TRAN_Calc_Orb2Real*/

static void TRAN_Current_dOrb(int NUM_c, int *MP,
  double ***JSym, double **JASym, double **JReal)
{
  int myid;
  int Mc_AN, Gc_AN;
  int h_AN, Gh_AN, Mh_AN, Rnh;
  int Nog, Nc, Nh;
  int i, j, GN, Rn;
  int iaxis;
  double x, y, z, Cxyz[4];
  double **dorbs0, tmp[3];

  dorbs0 = (double**)malloc(sizeof(double*) * 4);
  for (iaxis = 0; iaxis < 4; iaxis++)
    dorbs0[iaxis] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  /*
  Resiprocal fractiona coordinate -> Cartecian
  */
  for (i = 0; i < NUM_c; i++){
    for (j = 0; j < NUM_c; j++){
      for (iaxis = 0; iaxis < 2; iaxis++) tmp[iaxis] = JSym[iaxis][i][j];
      for (iaxis = 0; iaxis < 3; iaxis++)
        JSym[iaxis][i][j] = rtv[2][iaxis + 1] * tmp[0]
                          + rtv[3][iaxis + 1] * tmp[1];
    } /* for (i = 0; i < NUM_c; i++)*/
  } /* for (j = 0; j < NUM_c; j++)*/

  MPI_Comm_rank(mpi_comm_level1, &myid);

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Rnh = ncn[Gc_AN][h_AN];

      if (atv_ijk[Rnh][1] != 0) continue;

      for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++){

        Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
        Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
        GN = GridListAtom[Mc_AN][Nc];

        Rn = CellListAtom[Mc_AN][Nc];
        if (atv_ijk[Rn][1] != 0) continue;

        Get_Grid_XYZ(GN, Cxyz);
       /*
        Get derivatve at GN1, species Cwan1
        */
        x = Cxyz[1] + atv[Rn][1] - Gxyz[Gc_AN][1];
        y = Cxyz[2] + atv[Rn][2] - Gxyz[Gc_AN][2];
        z = Cxyz[3] + atv[Rn][3] - Gxyz[Gc_AN][3];

        Get_dOrbitals(WhatSpecies[Gc_AN], x, y, z, dorbs0);

        if (G2ID[Gh_AN] == myid){
          for (iaxis = 0; iaxis < 3; iaxis++){
            for (i = 0; i < Spe_Total_CNO[WhatSpecies[Gc_AN]]; i++){
              for (j = 0; j < Spe_Total_CNO[WhatSpecies[Gh_AN]]; j++){
                 JReal[iaxis][GN] +=
                  + JASym[MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                  * dorbs0[iaxis + 1][i] * Orbs_Grid[Mh_AN][Nh][j]
                  + JSym[iaxis][MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                  * Orbs_Grid[Mc_AN][Nc][i] * Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
              }/*for (j = 0; j < NO1; j++)*/
            }/*for (i = 0; i < NO0; i++)*/
          }/*(iaxis = 0; iaxis < 3; iaxis++)*/
        }/*if (G2ID[Gh_AN] == myid)*/
        else{
          for (iaxis = 0; iaxis < 3; iaxis++){
            for (i = 0; i < Spe_Total_CNO[WhatSpecies[Gc_AN]]; i++){
              for (j = 0; j < Spe_Total_CNO[WhatSpecies[Gh_AN]]; j++){
                JReal[iaxis][GN] +=
                  + JASym[MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                  * dorbs0[iaxis + 1][i] * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j]
                  + JSym[iaxis][MP[Gc_AN] + i - 1][MP[Gh_AN] + j - 1]
                  * Orbs_Grid[Mc_AN][Nc][i] * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */
              } /*for (j = 0; j < NO1; j++)*/
            } /*for (i = 0; i < NO0; i++)*/
          }/*(iaxis = 0; iaxis < 3; iaxis++)*/
        } /*if (G2ID[Gh_AN] != myid)*/

      } /*for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++)*/
    } /*for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)*/
  } /*for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)*/
  /*
  Cartecian -> Fractional coordinate
  */
  for (GN = 0; GN < TNumGrid; GN++){
    for (iaxis = 0; iaxis < 3; iaxis++) tmp[iaxis] = JReal[iaxis][GN];
    for (iaxis = 0; iaxis < 3; iaxis++)
      JReal[iaxis][GN] = (tv[iaxis + 1][1] * tmp[0]
                        + tv[iaxis + 1][2] * tmp[1]
                        + tv[iaxis + 1][3] * tmp[2]);
  } /*for (i1 = 0; i1 < TNumGrid; i1++)*/

  for (iaxis = 0; iaxis < 3; iaxis++)
    MPI_Allreduce(MPI_IN_PLACE, JReal[iaxis], TNumGrid, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (iaxis = 0; iaxis < 4; iaxis++) free(dorbs0[iaxis]);
  free(dorbs0);
} /*static void TRAN_Current_dOrb*/

static void TRAN_Integrate1D(double **Jmat, dcomplex **Jbound, double *Current)
{
  int i1, i2, i3, iside, myid;
  double Jloc, Jtot, length_tv1;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  length_tv1 = length_gtv[1] * Ngrid1;

  if (myid == Host_ID) printf("    Sum of current in real space [Ampere] \n");

  for (iside = 0; iside < 2; iside++){
    Jtot = 0.0;
    for (i1 = 0; i1 < Ngrid1; i1++){
      for (i2 = 0; i2 < Ngrid2; i2++){
        for (i3 = 0; i3 < Ngrid3; i3++){
          Jbound[iside][i3 + i2*Ngrid3].r = Jbound[iside][i3 + i2*Ngrid3].r
            + Jmat[iside][i3 + i2*Ngrid3 + i1*Ngrid3*Ngrid2] * length_gtv[1];
 
          Jtot += Jmat[iside][i3 + i2*Ngrid3 + i1*Ngrid3*Ngrid2];
        } /*for (i3 = 0; i3 < Ngrid3; i3++)*/
      } /*for (i2 = 0; i2 < Ngrid2; i2++)*/
    } /*for (i1 = 0; i1 < Ngrid1; i1++)*/

    Jtot *= 0.0066236178 * GridVol;
    if (myid == Host_ID) {
      if(iside == 0) printf("      Left: %15.5e\n", Jtot);
      else printf("      Right: %15.5e\n", Jtot);
    }

  } /*for (iside = 0; iside < 2; iside++)*/
  
  for (iside = 0; iside < 2; iside++){
    for (i2 = 0; i2 < Ngrid2; i2++){
      for (i3 = 0; i3 < Ngrid3; i3++){

        /*Fractional -> Cartesian*/
        Jloc = Current[i3 + i2*Ngrid3 + iside*(Ngrid1 - 1)*Ngrid3*Ngrid2] / length_tv1;

        /*Cartesian -> Fractional*/
        Current[i3 + i2*Ngrid3 + iside*(Ngrid1 - 1)*Ngrid3*Ngrid2] = 
          Jbound[iside][i3 + i2*Ngrid3].r * length_tv1;

        /*Cartesian*/
        Jbound[iside][i3 + i2*Ngrid3].r = Jbound[iside][i3 + i2*Ngrid3].r - Jloc;
        Jbound[iside][i3 + i2*Ngrid3].i = 0.0;
      } /*for (i3 = 0; i3 < Ngrid3; i3++)*/
    } /*for (i2 = 0; i2 < Ngrid2; i2++)*/
  } /*for (iside = 0; iside < 2; iside++)*/
} /* static void TRAN_Integrate1D */

static void TRAN_Current_NonLoc(double *Rho, double **Current)
{
  int iaxis, jaxis;
  int i1, i2, i3;
  int i1p, i2p, i3p;
  int i1m, i2m, i3m;
  double dfrac1, dfrac2, dfrac3, dcurrent[3][3], tmp[3];

  dfrac1 = 2.0 / (double)Ngrid1;
  dfrac2 = 2.0 / (double)Ngrid2;
  dfrac3 = 2.0 / (double)Ngrid3;

  /*Fractional coordinate*/

  for (i1 = 0; i1 < Ngrid1; i1++) {

    if (i1 == 0) i1m = i1;
    else i1m = (i1 - 1 + Ngrid1) % Ngrid1;

    if (i1 == Ngrid1 - 1) i1p = i1;
    else i1p = (i1 + 1 + Ngrid1) % Ngrid1;

    for (i2 = 0; i2 < Ngrid2; i2++) {
      i2m = (i2 - 1 + Ngrid2) % Ngrid2;
      i2p = (i2 + 1 + Ngrid2) % Ngrid2;
      for (i3 = 0; i3 < Ngrid3; i3++) {
        i3m = (i3 - 1 + Ngrid3) % Ngrid3;
        i3p = (i3 + 1 + Ngrid3) % Ngrid3;

        for (iaxis = 0; iaxis < 3; iaxis++) {

          if (i1 == 0 || i1 == Ngrid1 - 1) {
            dcurrent[iaxis][0] =
              (Current[iaxis][i3 + i2*Ngrid3 + i1p*Ngrid2*Ngrid3]
                - Current[iaxis][i3 + i2*Ngrid3 + i1m*Ngrid2*Ngrid3]) / (0.5 * dfrac1);
          }
          else {
            dcurrent[iaxis][0] =
              (Current[iaxis][i3 + i2*Ngrid3 + i1p*Ngrid2*Ngrid3]
                - Current[iaxis][i3 + i2*Ngrid3 + i1m*Ngrid2*Ngrid3]) / dfrac1;
          } /*if (i1 != 0 && i1 != Ngrid1 - 1)*/

          dcurrent[iaxis][1] =
            (Current[iaxis][i3 + i2p*Ngrid3 + i1*Ngrid2*Ngrid3]
              - Current[iaxis][i3 + i2m*Ngrid3 + i1*Ngrid2*Ngrid3]) / dfrac2;

          dcurrent[iaxis][2] =
            (Current[iaxis][i3p + i2*Ngrid3 + i1*Ngrid2*Ngrid3]
              - Current[iaxis][i3m + i2*Ngrid3 + i1*Ngrid2*Ngrid3]) / dfrac3;
        }
        /*
        Fractional coord. -> Cartecian
        */
        for (iaxis = 0; iaxis < 3; iaxis++) {
          for (jaxis = 0; jaxis < 3; jaxis++) tmp[jaxis] = dcurrent[iaxis][jaxis];
          for (jaxis = 0; jaxis < 3; jaxis++) {
            dcurrent[iaxis][jaxis] = (rtv[1][jaxis + 1] * tmp[0]
                                    + rtv[2][jaxis + 1] * tmp[1]
                                    + rtv[3][jaxis + 1] * tmp[2]) / (2.0 * PI);
          } /*for (jaxis = 0; jaxis < 3; jaxis++)*/
        }/*for (iaxis = 0; iaxis < 3; iaxis++)*/
   
        for (iaxis = 0; iaxis < 3; iaxis++) {
          for (jaxis = 0; jaxis < 3; jaxis++) tmp[jaxis] = dcurrent[jaxis][iaxis];
          for (jaxis = 0; jaxis < 3; jaxis++) {
            dcurrent[jaxis][iaxis] = (rtv[1][jaxis + 1] * tmp[0]
                                    + rtv[2][jaxis + 1] * tmp[1]
                                    + rtv[3][jaxis + 1] * tmp[2]) / (2.0 * PI);
          } /*for (jaxis = 0; jaxis < 3; jaxis++)*/
        }/*for (iaxis = 0; iaxis < 3; iaxis++)*/

        Rho[i3 + i2*Ngrid3 + i1*Ngrid2*Ngrid3] =
          -(dcurrent[0][0] + dcurrent[1][1] + dcurrent[2][2]);

      } /*for (i3 = 0; i3 < Ngrid3; i3++)*/
    } /*for (i2 = 0; i2 < Ngrid2; i2++)*/
  } /*for (i1 = 1; i1 < Ngrid1; i1++)*/

} /*static void TRAN_Current_NonLoc*/

static void TRAN_Current_Boundary(dcomplex **JBound)
{
  int k2, k3;
  int side;
  int n2, n3;
  double tmp;
  fftw_complex *in, *out;
  fftw_plan p;

  /* allocation of array */

  in = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);

  for (side = 0; side <= 1; side++){

    /* FFT of JBound for c-axis */

    p = fftw_plan_dft_1d(Ngrid3, in, out, -1, FFTW_ESTIMATE);

    for (n2 = 0; n2 < Ngrid2; n2++){

      for (n3 = 0; n3 < Ngrid3; n3++){

        in[n3][0] = JBound[side][n2 * Ngrid3 + n3].r;
        in[n3][1] = JBound[side][n2 * Ngrid3 + n3].i;
      } /*for (n3 = 0; n3 < Ngrid3; n3++)*/

      fftw_execute(p);

      for (k3 = 0; k3 < Ngrid3; k3++){

        JBound[side][n2 * Ngrid3 + k3].r = out[k3][0];
        JBound[side][n2 * Ngrid3 + k3].i = out[k3][1];
      } /*for (k3 = 0; k3 < Ngrid3; k3++)*/
    } /*for (n2 = 0; n2 < Ngrid2; n2++)*/

    fftw_destroy_plan(p);

    /* FFT of JBound for b-axis */

    p = fftw_plan_dft_1d(Ngrid2, in, out, -1, FFTW_ESTIMATE);

    for (k3 = 0; k3 < Ngrid3; k3++){

      for (n2 = 0; n2 < Ngrid2; n2++){

        in[n2][0] = JBound[side][n2 * Ngrid3 + k3].r;
        in[n2][1] = JBound[side][n2 * Ngrid3 + k3].i;
      } /*for (n2 = 0; n2 < Ngrid2; n2++)*/

      fftw_execute(p);

      for (k2 = 0; k2 < Ngrid2; k2++){

        JBound[side][k2 * Ngrid3 + k3].r = out[k2][0];
        JBound[side][k2 * Ngrid3 + k3].i = out[k2][1];
      } /*for (k2 = 0; k2 < Ngrid2; k2++)*/
    } /*for (k3 = 0; k3 < Ngrid3; k3++)*/

    fftw_destroy_plan(p);

    tmp = 1.0 / (double)(Ngrid2*Ngrid3);

    for (k2 = 0; k2 < Ngrid2; k2++){
      for (k3 = 0; k3 < Ngrid3; k3++){
        JBound[side][k2 * Ngrid3 + k3].r *= tmp;
        JBound[side][k2 * Ngrid3 + k3].i *= tmp;
      } /*for (k3 = 0; k3 < Ngrid3; k3++)*/
    } /*for (k2 = 0; k2 < Ngrid2; k2++)*/

  } /* side */

  /* freeing of arrays */

  fftw_free(in);
  fftw_free(out);
} /*static void TRAN_Current_Boundary*/

static void FFT2D_CurrentDensity(double *ReRhok, double *ImRhok)
{
  int BN_AB, BN_CB;
  double tmp0;
  double *ReRhor, *ImRhor;

  ReRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB);
  ImRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB);

  /* set ReRhor and ImRhor */

  for (BN_AB = 0; BN_AB < My_NumGridB_AB; BN_AB++){
    ReRhor[BN_AB] = ReRhok[BN_AB];
    ImRhor[BN_AB] = 0.0;
  } /* for (BN_AB = 0; BN_AB < My_NumGridB_AB; BN_AB++) */

  /****************************************************
  FFT of Dens
  ****************************************************/

  FFT2D_Poisson(ReRhor, ImRhor, ReRhok, ImRhok);

  tmp0 = 1.0 / (double)(Ngrid3*Ngrid2);
  for (BN_CB = 0; BN_CB<My_NumGridB_CB; BN_CB++){
    ReRhok[BN_CB] *= tmp0;
    ImRhok[BN_CB] *= tmp0;
  } /* for (BN_CB = 0; BN_CB<My_NumGridB_CB; BN_CB++) */

  /* freeing of arrays */

  free(ReRhor);
  free(ImRhor);
} /*static void FFT2D_CurrentDensity*/

static void TRAN_Poisson_Current(double *Rho, dcomplex **Jbound)
{
  int i, k1, k2, k3;
  int GN, GNs, BN_CB, N2D_12, N2D_23;
  double sk2, sk3, Gx, Gy, Gz;
  double da2, Gpara2;
  int myid, numprocs;
  dcomplex *DL, *D, *DU, *B;
  INTEGER n, nrhs, ldb, info;
  double *ReRhok, *ImRhok;

  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  MPI_Barrier(mpi_comm_level1);

  /****************************************************
  allocation of arrays
  ****************************************************/

  DL = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  D = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  DU = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  B = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);

  ReRhok = (double*)malloc(sizeof(double)*My_Max_NumGridB);
  ImRhok = (double*)malloc(sizeof(double)*My_Max_NumGridB);
  N2D_12 = Ngrid1*Ngrid2;
  for (i = 0; i < My_Max_NumGridB; i++){
    ReRhok[i] = 0.0;
    ImRhok[i] = 0.0;
  }
  for (i = 0; i < My_NumGridB_AB; i++){
    ReRhok[i] = Rho[i + ((myid*N2D_12 + numprocs - 1) / numprocs)*Ngrid3];
  }
  for (i = 0; i < TNumGrid; i++) Rho[i] = 0.0;

  /****************************************************
  FFT of charge density on the b-c plane
  ****************************************************/

  FFT2D_CurrentDensity(ReRhok, ImRhok);

  /****************************************************
  solve finite difference equations
  ****************************************************/

  da2 = Dot_Product(gtv[1], gtv[1]);
  N2D_23 = Ngrid3*Ngrid2;
  GNs = ((myid*N2D_23 + numprocs - 1) / numprocs)*Ngrid1;

  for (BN_CB = 0; BN_CB<My_NumGridB_CB; BN_CB += Ngrid1){

    GN = BN_CB + GNs;
    k3 = GN / (Ngrid2*Ngrid1);
    k2 = (GN - k3*Ngrid2*Ngrid1) / Ngrid1;

    if (k2<Ngrid2 / 2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3 / 2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk2*rtv[2][2] + sk3*rtv[3][2];
    Gz = sk2*rtv[2][3] + sk3*rtv[3][3];
    Gpara2 = Gx*Gx + Gy*Gy + Gz*Gz;

    for (k1 = 0; k1<(Ngrid1 - 1); k1++){
      DL[k1].r = 1.0;
      DL[k1].i = 0.0;
      DU[k1].r = 1.0;
      DU[k1].i = 0.0;
    }

    for (k1 = 0; k1<(Ngrid1 - 0); k1++){
      D[k1].r = -2.0 - da2*Gpara2;
      D[k1].i = 0.0;
    }

    /* scale terms with Gpara=0 */

    if (k2 == 0 && k3 == 0){
      for (k1 = 0; k1<Ngrid1; k1++){
        ReRhok[BN_CB + k1] *= TRAN_Poisson_Gpara_Scaling;
      }
    }

    /* set B */

    for (k1 = 0; k1<(Ngrid1 - 0); k1++){
      B[k1].r = -da2*ReRhok[BN_CB + k1];
      B[k1].i = -da2*ImRhok[BN_CB + k1];
    }

    /* add the boundary condition */
    B[0].r = B[0].r - 2.0 * sqrt(da2) * Jbound[0][k3 + k2*Ngrid3].r;
    B[0].i = B[0].i - 2.0 * sqrt(da2) * Jbound[0][k3 + k2*Ngrid3].i;
    B[Ngrid1 - 1].r = B[Ngrid1 - 1].r + 2.0 * sqrt(da2) * Jbound[1][k3 + k2*Ngrid3].r;
    B[Ngrid1 - 1].i = B[Ngrid1 - 1].i + 2.0 * sqrt(da2) * Jbound[1][k3 + k2*Ngrid3].i;
    DL[Ngrid1 - 2].r = 2.0;
    DU[0].r = 2.0;

    /* solve the linear equation */

    if (k2 == 0 && k3 == 0) {
      ReRhok[BN_CB] = 0.0;
      ReRhok[BN_CB + 1] = (B[0].r - D[0].r * ReRhok[BN_CB]) / DU[0].r;
      for (k1 = 1; k1 < Ngrid1 - 1; k1++) {
        ReRhok[BN_CB + k1 + 1] = (B[k1].r
          - DL[k1 - 1].r * ReRhok[BN_CB + k1 - 1]
           - D[k1    ].r * ReRhok[BN_CB + k1    ]) / DU[k1].r;
      }
      for (k1 = 0; k1 < Ngrid1; k1++) {
        ImRhok[BN_CB + k1] = 0;
      }
      /*printf("DEBUG5:  %15.5e  %15.5e\n", DL[Ngrid1 - 2].r * ReRhok[Ngrid1 - 2] + D[Ngrid1 - 1].r * ReRhok[Ngrid1 - 1], B[Ngrid1 - 1].r);*/
    }
    else {
      n = Ngrid1 - 0;
      nrhs = 1;
      ldb = Ngrid1 - 0;

      F77_NAME(zgtsv, ZGTSV)(&n, &nrhs, DL, D, DU, B, &ldb, &info);

      /* store B to ReRhok and ImRhok */

      for (k1 = 0; k1<(Ngrid1 - 0); k1++) {
        ReRhok[BN_CB + k1] = B[k1].r;
        ImRhok[BN_CB + k1] = B[k1].i;
      }
    }

  } /* BN2 */

  /****************************************************
  find the Hartree potential in real space
  ****************************************************/

  Get_Value_inReal2D(0, &Rho[((myid*N2D_12 + numprocs - 1) / numprocs)*Ngrid3], 
    NULL, ReRhok, ImRhok);
  MPI_Allreduce(MPI_IN_PLACE, Rho, TNumGrid, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
  freeing of arrays
  ****************************************************/

  free(DL);
  free(D);
  free(DU);
  free(B);
  free(ReRhok);
  free(ImRhok);
}

static void TRAN_Current_AddNLoc(double *Rho, double **Current)
{
  int i1, i2, i3;
  int i1p, i2p, i3p;
  int i1m, i2m, i3m;
  double dfrac1, dfrac2, dfrac3;
  int myid;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  for (i1 = 0; i1 < Ngrid1; i1++) {/*DEBUG*/
    dfrac1 = 0.0;/*DEBUG*/
    for (i2 = 0; i2 < Ngrid2*Ngrid3; i2++)/*DEBUG*/
      dfrac1 += Current[0][i2 + i1*Ngrid2*Ngrid3] / (length_gtv[1] * Ngrid1);/*DEBUG*/
    dfrac1 *= 0.0066236178 * GridVol / length_gtv[1];/*DEBUG*/
    /*if(myid == Host_ID) printf("DEBUG2: %d %15.7e\n", i1, dfrac1);*/
  }/*DEBUG*/
  /*if (myid == Host_ID) printf("DEBUG2: \n");*/

  dfrac1 = 2.0 / (double)Ngrid1;
  dfrac2 = 2.0 / (double)Ngrid2;
  dfrac3 = 2.0 / (double)Ngrid3;

  /*Fractional coordinate*/

  for (i1 = 0; i1 < Ngrid1; i1++){
    i1m = (i1 - 1 + Ngrid1) % Ngrid1;
    i1p = (i1 + 1 + Ngrid1) % Ngrid1;
    for (i2 = 0; i2 < Ngrid2; i2++){
      i2m = (i2 - 1 + Ngrid2) % Ngrid2;
      i2p = (i2 + 1 + Ngrid2) % Ngrid2;
      for (i3 = 0; i3 < Ngrid3; i3++){
        i3m = (i3 - 1 + Ngrid3) % Ngrid3;
        i3p = (i3 + 1 + Ngrid3) % Ngrid3;

        if (i1 != 0 && i1 != Ngrid1 - 1){
          Current[0][i3 + i2*Ngrid3 + i1*Ngrid2*Ngrid3] -=
            ( Rho[i3 + i2*Ngrid3 + i1p*Ngrid2*Ngrid3]
            - Rho[i3 + i2*Ngrid3 + i1m*Ngrid2*Ngrid3]) / dfrac1;
        } /*if (i1 != 0 && i1 != Ngrid1 - 1)*/

        Current[1][i3 + i2*Ngrid3 + i1*Ngrid2*Ngrid3] -=
          ( Rho[i3 + i2p*Ngrid3 + i1*Ngrid2*Ngrid3]
          - Rho[i3 + i2m*Ngrid3 + i1*Ngrid2*Ngrid3]) / dfrac2;

        Current[2][i3 + i2*Ngrid3 + i1*Ngrid2*Ngrid3] -=
          ( Rho[i3p + i2*Ngrid3 + i1*Ngrid2*Ngrid3]
          - Rho[i3m + i2*Ngrid3 + i1*Ngrid2*Ngrid3]) / dfrac3;
      } /*for (i3 = 0; i3 < Ngrid3; i3++)*/
    } /*for (i2 = 0; i2 < Ngrid2; i2++)*/
  } /*for (i1 = 1; i1 < Ngrid1; i1++)*/

  for (i1 = 0; i1 < Ngrid1; i1++) {/*DEBUG*/
    dfrac1 = 0.0;/*DEBUG*/
    for (i2 = 0; i2 < Ngrid2*Ngrid3; i2++)/*DEBUG*/
      dfrac1 += Current[0][i2 + i1*Ngrid2*Ngrid3] / (length_gtv[1] * Ngrid1);/*DEBUG*/
    dfrac1 *= 0.0066236178 * GridVol / length_gtv[1];/*DEBUG*/
    /*if (myid == Host_ID) printf("DEBUG2: %d  %15.7e\n", i1, dfrac1);*/
  }/*DEBUG*/

} /*static void TRAN_Current_AddNLoc*/

static void TRAN_Current_Spin(double ***CDensity){
  int iaxis, i;
  double CDensity0;

  if (SpinP_switch == 0){
    for (iaxis = 0; iaxis < 3; iaxis++){
      for (i = 0; i < TNumGrid; i++) CDensity[0][iaxis][i] = 2.0 * CDensity[0][iaxis][i];
    } /*for (iaxis = 0; iaxis < 3; iaxis++)*/
  } /*if (SpinP_switch == 0)*/
  else{
    for (iaxis = 0; iaxis < 3; iaxis++){
      for (i = 0; i < TNumGrid; i++){
        CDensity0 = CDensity[0][iaxis][i] + CDensity[1][iaxis][i];
        CDensity[1][iaxis][i] = CDensity[0][iaxis][i] - CDensity[1][iaxis][i];
        CDensity[0][iaxis][i] = CDensity0;
      } /*for (i = 0; i < TNumGrid; i++)*/
    } /* for (iaxis = 0; iaxis < 3; iaxis++) */
  } /*if (SpinP_switch != 0)*/
} /*static void TRAN_Current_Spin*/

static void Print_CubeTitle_CDensity(FILE *fp, int ispin)
{
  int ct_AN;
  int spe;

  if (ispin == 0) fprintf(fp, " Charge Current Density \n");
  else fprintf(fp, " Spin Current Density \n");
  fprintf(fp, " The component parallel to the a-axis \n");
 
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    atomnum, Grid_Origin[1], Grid_Origin[2], Grid_Origin[3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid1, gtv[1][1], gtv[1][2], gtv[1][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid2, gtv[2][1], gtv[2][2], gtv[2][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid3, gtv[3][1], gtv[3][2], gtv[3][3]);

  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
    spe = WhatSpecies[ct_AN];
    fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
      Spe_WhatAtom[spe],
      Spe_Core_Charge[spe] - InitN_USpin[ct_AN] - InitN_DSpin[ct_AN],
      Gxyz[ct_AN][1], Gxyz[ct_AN][2], Gxyz[ct_AN][3]);
  } /* for (ct_AN = 1; ct_AN <= atomnum; ct_AN++) */

} /* static void Print_CubeTitle_CDensity */

static void Print_CubeTitle_CDensity_Bin(FILE *fp, int ispin)
{
  int ct_AN, n;
  int spe;
  char clist[300];
  double *dlist;

  if (ispin == 0) sprintf(clist, " Charge Current Density \n");
  else sprintf(clist, " Spin Current Density \n");
  fwrite(clist, sizeof(char), 200, fp);

  sprintf(clist, " The component parallel to the a-axis \n");
  fwrite(clist, sizeof(char), 200, fp);

  fwrite(&atomnum, sizeof(int), 1, fp);
  fwrite(&Grid_Origin[1], sizeof(double), 3, fp);
  fwrite(&Ngrid1, sizeof(int), 1, fp);
  fwrite(&gtv[1][1], sizeof(double), 3, fp);
  fwrite(&Ngrid2, sizeof(int), 1, fp);
  fwrite(&gtv[2][1], sizeof(double), 3, fp);
  fwrite(&Ngrid3, sizeof(int), 1, fp);
  fwrite(&gtv[3][1], sizeof(double), 3, fp);

  dlist = (double*)malloc(sizeof(double)*atomnum * 5);

  n = 0;
  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
    spe = WhatSpecies[ct_AN];
    dlist[n] = (double)Spe_WhatAtom[spe];                                  n++;
    dlist[n] = Spe_Core_Charge[spe] - InitN_USpin[ct_AN] - InitN_DSpin[ct_AN]; n++;
    dlist[n] = Gxyz[ct_AN][1];                                             n++;
    dlist[n] = Gxyz[ct_AN][2];                                             n++;
    dlist[n] = Gxyz[ct_AN][3];                                             n++;
  }

  fwrite(dlist, sizeof(double), atomnum * 5, fp);
  free(dlist);
}/* static void Print_CubeTitle_Bin */


static void Print_CubeCData_CDensity(
  FILE *fp,
  double *data)
{
  int i1, i2, i3;
  int GN;

  for (i1 = 0; i1<Ngrid1; i1++){
    for (i2 = 0; i2<Ngrid2; i2++){
      for (i3 = 0; i3<Ngrid3; i3++){
        GN = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;
        fprintf(fp, "%13.3E", data[GN]);
        if ((i3 + 1) % 6 == 0) { fprintf(fp, "\n"); }
      }
      /* avoid double \n\n when Ngrid3%6 == 0  */
      if (Ngrid3 % 6 != 0) fprintf(fp, "\n");
    }
  }
} /* static void Print_CubeCData_MO */

static void TRAN_Current_OutPutCube(double ***CDensity, int TRAN_OffDiagonalCurrent)
{
  int fd, ispin, igrid;
  FILE *fp;
  int myid;
  char file1[YOUSO10], Out_Extention[YOUSO10], write_mode[YOUSO10];
  double length_tv1;
  double *CDensity0;
  char buf[fp_bsize];          /* setvbuf */

  length_tv1 = length_gtv[1] * Ngrid1;
  CDensity0 = (double*)malloc(sizeof(double)*(TNumGrid));

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID){

    if (OutData_bin_flag){
      sprintf(Out_Extention, ".cube.bin");
      sprintf(write_mode, "wb");
    } /*if (OutData_bin_flag)*/
    else{
      sprintf(Out_Extention, ".cube");
      sprintf(write_mode, "w");
    }

    for (ispin = 0; ispin < SpinP_switch + 1; ispin++){

      if (ispin == 0) {
        sprintf(file1, "%s%s.curden1%s", filepath, filename, Out_Extention);
        printf("  Charge-current density along a-axis: %s\n", file1);
      }
      else if (ispin == 1) {
        sprintf(file1, "%s%s.scurden1%s", filepath, filename, Out_Extention);
        printf("  Spin-current density along a-axis: %s\n", file1);
      }
      else if (ispin == 2 && TRAN_OffDiagonalCurrent == 1) {
        sprintf(file1, "%s%s.odcurden1_r%s", filepath, filename, Out_Extention);
        printf("  Off-diagonal current density along a-axis (real): %s\n", file1);
      }
      else if (ispin == 3 && TRAN_OffDiagonalCurrent == 1) {
        sprintf(file1, "%s%s.odcurden1_i%s", filepath, filename, Out_Extention);
        printf("  Off-diagonal current density along a-axis (imaginal): %s\n", file1);
      }

      if ((fp = fopen(file1, write_mode)) != NULL){

#ifdef xt3
        setvbuf(fp, buf, _IOFBF, fp_bsize);  /* setvbuf */
#endif

        for (igrid = 0; igrid < TNumGrid; igrid++) 
          CDensity0[igrid] = CDensity[ispin][0][igrid] / length_tv1;

        if (OutData_bin_flag){
          Print_CubeTitle_CDensity_Bin(fp, ispin);
          fwrite(CDensity0, sizeof(double), Ngrid1 * Ngrid2 * Ngrid3, fp);
        }
        else{
          Print_CubeTitle_CDensity(fp, ispin);
          Print_CubeCData_CDensity(fp, CDensity0);
          fd = fileno(fp);
          fsync(fd);
        }

        fclose(fp);

      } /*if ((fp = fopen(file1, write_mode)) != NULL)*/
      else{
        printf("Failure of saving CurrentDensity\n");
      }
    } /*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/
  } /* if (myid == Host_ID) */

  free(CDensity0);

} /*static void TRAN_Current_OutPutCube*/

static void Print_VectorData_CDensity(
  double **CDensity, int ispin, int numcur, int *curindx, 
  int TRAN_OffDiagonalCurrent)
{
  FILE *fp;
  int i, j, k, GN;
  double Cxyz[4];
  int myid;
  char file1[YOUSO10];

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID){
    
    if (ispin == 0) {
      sprintf(file1, "%s%s.curden.xsf", filepath, filename);
      printf("  Charge-current density: %s\n", file1);
    }
    else if (ispin == 1) {
      sprintf(file1, "%s%s.scurden.xsf", filepath, filename);
      printf("  Spin-current density: %s\n", file1);
    }
    else if (ispin == 2 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_r.xsf", filepath, filename);
      printf("  Off-diagonal current density (real): %s\n", file1);
    }
    else if (ispin == 3 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_i.xsf", filepath, filename);
      printf("  Off-diagonal current density (imaginal): %s\n", file1);
    }

    if ((fp = fopen(file1, "w")) != NULL){
      fprintf(fp, "CRYSTAL\n");
      fprintf(fp, "PRIMVEC\n");
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[1][1], BohrR*tv[1][2], BohrR*tv[1][3]);
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[2][1], BohrR*tv[2][2], BohrR*tv[2][3]);
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[3][1], BohrR*tv[3][2], BohrR*tv[3][3]);

      fprintf(fp, "PRIMCOORD\n");
      fprintf(fp, "%4d 1\n", atomnum + numcur);

      for (k = 1; k <= atomnum; k++){
        i = WhatSpecies[k];
        j = Spe_WhatAtom[i];
        fprintf(fp, "%s   %8.5f  %8.5f  %8.5f   0.0 0.0 0.0\n",
          Atom_Symbol[j],
          Gxyz[k][1] * BohrR,
          Gxyz[k][2] * BohrR,
          Gxyz[k][3] * BohrR);
      }

    /****************************************************
    fprintf vector data
    ****************************************************/

      for (i = 0; i < numcur; i++){
        GN = curindx[i];
        Get_Grid_XYZ(GN, Cxyz);
        fprintf(fp, "X %13.3E %13.3E %13.3E %13.3E %13.3E %13.3E\n",
          BohrR*Cxyz[1], BohrR*Cxyz[2], BohrR*Cxyz[3],
          CDensity[0][GN], CDensity[1][GN], CDensity[2][GN]);
      }

      fclose(fp);
    } /*if ((fp = fopen(file1, "w")) != NULL)*/
    else{
      printf("Failure of saving CurrentDensity\n");
    }
  } /*if (myid == Host_ID)*/
} /*static void Print_VectorData_CDensity*/

static void Print_VectorData_CDensity_bin(
  double **CDensity, int ispin, int numcur, int *curindx,
  int TRAN_OffDiagonalCurrent)
{
  FILE *fp;
  int i, j, k, n;
  int GN;
  double Cxyz[4], tv0[4][4];
  int myid;
  char file1[YOUSO10];
  char clist[300];
  double *dlist;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID){

    if (ispin == 0) {
      sprintf(file1, "%s%s.curden.xsf.bin", filepath, filename);
      printf("  Charge-current density: %s\n", file1);
    }
    else if (ispin == 1) {
      sprintf(file1, "%s%s.scurden.xsf.bin", filepath, filename);
      printf("  Spin-current density: %s\n", file1);
    }
    else if (ispin == 2 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_r.xsf.bin", filepath, filename);
      printf("  Off-diagonal current density (real): %s\n", file1);
    }
    else if (ispin == 3 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_i.xsf.bin", filepath, filename);
      printf("  Off-diagonal current density (imaginal): %s\n", file1);
    }

    if ((fp = fopen(file1, "wb")) != NULL){

      sprintf(clist, "CRYSTAL\n");
      fwrite(clist, sizeof(char), 200, fp);
      sprintf(clist, "PRIMVEC\n");
      fwrite(clist, sizeof(char), 200, fp);

      for (i = 1; i <= 3; i++){
        for (j = 1; j <= 3; j++){
          tv0[i][j] = BohrR*tv[i][j];
        }
      }

      fwrite(&tv0[1][1], sizeof(double), 3, fp);
      fwrite(&tv0[2][1], sizeof(double), 3, fp);
      fwrite(&tv0[3][1], sizeof(double), 3, fp);

      fwrite(&atomnum, sizeof(int), 1, fp);
      fwrite(&numcur, sizeof(int), 1, fp);

      for (k = 1; k <= atomnum; k++){
        i = WhatSpecies[k];
        j = Spe_WhatAtom[i];

        fwrite(Atom_Symbol[j], sizeof(char), 4, fp);

        Cxyz[1] = Gxyz[k][1] * BohrR;
        Cxyz[2] = Gxyz[k][2] * BohrR;
        Cxyz[3] = Gxyz[k][3] * BohrR;

        fwrite(&Cxyz[1], sizeof(double), 3, fp);
      }

      /****************************************************
      fprintf vector data
      ****************************************************/

      dlist = (double*)malloc(sizeof(double)*My_NumGridB_AB * 6);

      n = 0;
      for (i = 0; i < numcur; i++){
        GN = curindx[i];
        Get_Grid_XYZ(GN, Cxyz);

        dlist[n] = BohrR*Cxyz[1];
        dlist[n + 1] = BohrR*Cxyz[2];
        dlist[n + 2] = BohrR*Cxyz[3];
        dlist[n + 3] = CDensity[0][GN];
        dlist[n + 4] = CDensity[1][GN];
        dlist[n + 5] = CDensity[2][GN];

        n += 6;
      } /* for (i1 = 0; i1 < Ngrid1; i1 += 4)*/
      fwrite(dlist, sizeof(double), n, fp);

      fclose(fp);
      free(dlist);
    } /*if ((fp = fopen(file1, "wb")) != NULL)*/
    else{
      printf("Failure of saving CurrentDensity\n");
    }
  } /*if (myid == Host_ID)*/
} /*static void Print_VectorData_CDensity_bin*/

static void TRAN_Current_OutputVector(double ***CDensity, 
  int TRAN_OffDiagonalCurrent)
{
  int ispin, i, iaxis, numcur, i1, i2, i3, GN, interval;
  int *curindx;
  double tmp[3], avecur, avecur0, maxcur;

  interval = 4;
  curindx = (int*)malloc(sizeof(int)*TNumGrid);

  for (ispin = 0; ispin < SpinP_switch + 1; ispin++){
    /*
    Fractional coord. -> Cartecian
    */
    for (i = 0; i < TNumGrid; i++){
      for (iaxis = 0; iaxis < 3; iaxis++) tmp[iaxis] = CDensity[ispin][iaxis][i];
      for (iaxis = 0; iaxis < 3; iaxis++) {
        CDensity[ispin][iaxis][i] = (rtv[1][iaxis + 1] * tmp[0]
                                   + rtv[2][iaxis + 1] * tmp[1]
                                   + rtv[3][iaxis + 1] * tmp[2]) / (2.0 * PI);
      } /*for (iaxis = 0; iaxis < 3; iaxis++)*/

    } /*for (i = 0; i < TNumGrid; i++)*/

    maxcur = 0.0;
    avecur = 0.0;
    numcur = 0;
    for (i1 = 1; i1 < Ngrid1 - 1; i1 += interval){
      for (i2 = 1; i2 < Ngrid2 - 1; i2 += interval){
        for (i3 = 1; i3 < Ngrid3 - 1; i3 += interval){
          GN = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;
          avecur0 = sqrt(CDensity[ispin][0][GN] * CDensity[ispin][0][GN]
                       + CDensity[ispin][1][GN] * CDensity[ispin][1][GN]
                       + CDensity[ispin][2][GN] * CDensity[ispin][2][GN]);
          avecur += avecur0;
          if (avecur0 > maxcur) maxcur = avecur0;
          numcur++;
        }
      }
    }

    avecur /= (double)(1 * numcur);
    maxcur *= 0.1;
    numcur = 0;
    for (i1 = 1; i1 < Ngrid1 - 1; i1 += interval){
      for (i2 = 1; i2 < Ngrid2 - 1; i2 += interval){
        for (i3 = 1; i3 < Ngrid3 - 1; i3 += interval){
          GN = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;
          avecur0 = sqrt( CDensity[ispin][0][GN] * CDensity[ispin][0][GN]
                        + CDensity[ispin][1][GN] * CDensity[ispin][1][GN]
                        + CDensity[ispin][2][GN] * CDensity[ispin][2][GN]);
          if (avecur0 > maxcur) {
            curindx[numcur] = GN;
            numcur++;
          }
        }
      }
    }

    if (OutData_bin_flag){
      Print_VectorData_CDensity_bin(CDensity[ispin], ispin, numcur, curindx,
        TRAN_OffDiagonalCurrent);
    }
    else{
      Print_VectorData_CDensity(CDensity[ispin], ispin, numcur, curindx,
        TRAN_OffDiagonalCurrent);
    }
  } /*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/

  free(curindx);

} /*static void TRAN_Current_OutputVector*/

static void Print_Voronoi_CDensity(
  double **VCurrent, int ispin,
  int TRAN_OffDiagonalCurrent)
{
  FILE *fp;
  int i, j, k;
  int myid;
  char file1[YOUSO10];

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID) {

    if (ispin == 0) {
      sprintf(file1, "%s%s.curden_atom.xsf", filepath, filename);
      printf("  Voronoi Charge-current density: %s\n", file1);
    }
    else if (ispin == 1) {
      sprintf(file1, "%s%s.scurden_atom.xsf", filepath, filename);
      printf("  Voronoi Spin-current density: %s\n", file1);
    }
    else if (ispin == 2 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_atom_r.xsf", filepath, filename);
      printf("  Voronoi Off-diagonal current density (real): %s\n", file1);
    }
    else if (ispin == 3 && TRAN_OffDiagonalCurrent == 1) {
      sprintf(file1, "%s%s.odcurden_atom_i.xsf", filepath, filename);
      printf("  Voronoi Off-diagonal current density (imaginal): %s\n", file1);
    }

    if ((fp = fopen(file1, "w")) != NULL) {
      fprintf(fp, "CRYSTAL\n");
      fprintf(fp, "PRIMVEC\n");
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[1][1], BohrR*tv[1][2], BohrR*tv[1][3]);
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[2][1], BohrR*tv[2][2], BohrR*tv[2][3]);
      fprintf(fp, " %10.6f %10.6f %10.6f\n", BohrR*tv[3][1], BohrR*tv[3][2], BohrR*tv[3][3]);

      fprintf(fp, "PRIMCOORD\n");
      fprintf(fp, "%4d 1\n", atomnum);

      for (k = 1; k <= atomnum; k++) {
        i = WhatSpecies[k];
        j = Spe_WhatAtom[i];
        fprintf(fp, "%s   %8.5f  %8.5f  %8.5f  %13.3e  %13.3e  %13.3e\n",
          Atom_Symbol[j],
          Gxyz[k][1] * BohrR, Gxyz[k][2] * BohrR, Gxyz[k][3] * BohrR, 
          VCurrent[k][0], VCurrent[k][1], VCurrent[k][2]);
      }

      fclose(fp);
    } /*if ((fp = fopen(file1, "w")) != NULL)*/
    else {
      printf("Failure of saving CurrentDensity\n");
    }
  } /*if (myid == Host_ID)*/
} /*static void Print_Voronoi_CDensity*/

static void TRAN_Voronoi_CDEN(double ***CDensity,
  int TRAN_OffDiagonalCurrent)
{
  int Mc_AN, Gc_AN, Nog, GNc, GRc, ispin, iaxis;
  double Cxyz[4], x, y, z, FuzzyW;
  double ***VCurrent, *Voronoi_Vol;

  VCurrent = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
  for (ispin = 0; ispin < SpinP_switch + 1; ispin++) {
    VCurrent[ispin] = (double**)malloc(sizeof(double*)*(atomnum + 1));
    for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++) {
      VCurrent[ispin][Gc_AN] = (double*)malloc(sizeof(double) * 3);
      for (iaxis = 0; iaxis < 3; iaxis++) VCurrent[ispin][Gc_AN][iaxis] = 0.0;
    }/*for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++)*/
  }/*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/

  Voronoi_Vol = (double*)malloc(sizeof(double)*(atomnum + 1));
  for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++) Voronoi_Vol[Gc_AN] = 0.0;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

    Gc_AN = M2G[Mc_AN];

    for (Nog = 0; Nog<NumOLG[Mc_AN][0]; Nog++) {

      /* calculate fuzzy weight */

      GNc = GridListAtom[Mc_AN][Nog];
      GRc = CellListAtom[Mc_AN][Nog];

      Get_Grid_XYZ(GNc, Cxyz);
      x = Cxyz[1] + atv[GRc][1];
      y = Cxyz[2] + atv[GRc][2];
      z = Cxyz[3] + atv[GRc][3];
      FuzzyW = Fuzzy_Weight(Gc_AN, Mc_AN, 0, x, y, z);

      for (ispin = 0; ispin < SpinP_switch + 1; ispin++) {
        for (iaxis = 0; iaxis < 3; iaxis++) {
          VCurrent[ispin][Gc_AN][iaxis] += FuzzyW * CDensity[ispin][iaxis][GNc] * GridVol;
        }/*for (iaxis = 0; iaxis < 3; iaxis++)*/
      }/*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/

      Voronoi_Vol[Gc_AN] += FuzzyW*GridVol*BohrR*BohrR*BohrR;

    }/*for (Nog = 0; Nog<NumOLG[Mc_AN][0]; Nog++)*/

  } /* Mc_AN */

  for (ispin = 0; ispin < SpinP_switch + 1; ispin++) {
    for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++) {
      MPI_Allreduce(MPI_IN_PLACE, VCurrent[ispin][Gc_AN], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }/*for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++)*/
  }/*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/
  MPI_Allreduce(MPI_IN_PLACE, Voronoi_Vol, atomnum + 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (ispin = 0; ispin < SpinP_switch + 1; ispin++) {
    for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++) {
      for (iaxis = 0; iaxis < 3; iaxis++) 
        VCurrent[ispin][Gc_AN][iaxis] /= Voronoi_Vol[Gc_AN];
    }/*for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++)*/

    Print_Voronoi_CDensity(VCurrent[ispin], ispin, TRAN_OffDiagonalCurrent);

  }/*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/

  for (ispin = 0; ispin < SpinP_switch + 1; ispin++) {
    for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++) {
      free(VCurrent[ispin][Gc_AN]);
    }/*for (Gc_AN = 1; Gc_AN < atomnum + 1; Gc_AN++)*/
    free(VCurrent[ispin]);
  }/*for (ispin = 0; ispin < SpinP_switch + 1; ispin++)*/
  free(VCurrent);

  free(Voronoi_Vol);

}/*void TRAN_Voronoi_CDEN*/

void TRAN_Output_AveCdensity(double ***CDensity)
{
  FILE *fp;
  int myid, GN, iaxis;
  char file1[YOUSO10];
  double MaxJ0, MaxJ;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID) {
    sprintf(file1, "%s%s.paranegf", filepath, filename);

    if ((fp = fopen(file1, "a")) != NULL) {

      fprintf(fp, "\nMaximum currentdensity [a.u.]\n\n");

      MaxJ = 0.0;
      for (GN = 0; GN < TNumGrid; GN++) {
        MaxJ0 = 0.0;
        for (iaxis = 0; iaxis < 3; iaxis++)
          MaxJ0 += CDensity[0][iaxis][GN] * CDensity[0][iaxis][GN];
        MaxJ0 = sqrt(MaxJ0);

        if (MaxJ0 > MaxJ)MaxJ = MaxJ0;
      }/* for (GN = 0; GN < TNumGrid; GN++) */

      fprintf(fp, "  Max.Currentdensity   %20.10e\n", MaxJ);

      fclose(fp);

    }/*if ((fp = fopen(file1, "a")) != NULL)*/
    else {
      printf("Failure of saving MOs\n");
    }

  }/* if (myid == Host_ID) */
}

void TRAN_CDen_Main(
  int NUM_c,
  int *MP,
  double ****JLocSym,
  double ***JLocASym,
  double ***Rho,
  double ****Jmat,
  dcomplex *SCC,
  int TRAN_OffDiagonalCurrent)
{
  int myid;
  double ***CDensity;
  double *RRho;
  double **RJmat;
  int ispin, iaxis, i, iside;
  dcomplex **Jbound;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID) printf("Start Calculation of the currentdensity\n\n");

  /*
  Allocate and initialize array
  */
  CDensity = (double***)malloc(sizeof(double**)*(SpinP_switch + 1));
  for (ispin = 0; ispin <= SpinP_switch; ispin++){
    CDensity[ispin] = (double**)malloc(sizeof(double*)*3);
    for (iaxis = 0; iaxis < 3; iaxis++){
      CDensity[ispin][iaxis] = (double*)malloc(sizeof(double) * TNumGrid);
      for (i = 0; i < TNumGrid; i++){
        CDensity[ispin][iaxis][i] = 0.0;
      }
    }
  }

  RRho = (double*)malloc(sizeof(double)*(TNumGrid));

  RJmat = (double**)malloc(sizeof(double*)*2);
  Jbound = (dcomplex**)malloc(sizeof(dcomplex*) * 2);
  for (iside = 0; iside < 2; iside++){
    RJmat[iside] = (double*)malloc(sizeof(double)*(TNumGrid));
    Jbound[iside] = (dcomplex*)malloc(sizeof(dcomplex)*(Ngrid2*Ngrid3));
  }
  /*
  */
  for (ispin = 0; ispin <= SpinP_switch; ispin++){
    if (myid == Host_ID) printf("  Spin #%d\n", ispin);

    for (i = 0; i < TNumGrid; i++) RRho[i] = 0.0;

    for (iside = 0; iside < 2; iside++){
      for (i = 0; i < TNumGrid; i++) RJmat[iside][i] = 0.0;
      for (i = 0; i < Ngrid2*Ngrid3; i++) {
        Jbound[iside][i].r = 0.0;
        Jbound[iside][i].i = 0.0;
      }
    }

    TRAN_Calc_Orb2Real(NUM_c, MP, Rho[ispin], RRho, SCC);

    TRAN_Calc_Orb2Real(NUM_c, MP, Jmat[ispin][0], RJmat[0], SCC);
    TRAN_Calc_Orb2Real(NUM_c, MP, Jmat[ispin][1], RJmat[1], SCC);

    TRAN_Current_dOrb(NUM_c, MP, JLocSym[ispin], 
      JLocASym[ispin], CDensity[ispin]);
    TRAN_Current_NonLoc(RRho, CDensity[ispin]);

    TRAN_Integrate1D(RJmat, Jbound, CDensity[ispin][0]);
    TRAN_Current_Boundary(Jbound);
    TRAN_Poisson_Current(RRho, Jbound);
    TRAN_Current_AddNLoc(RRho, CDensity[ispin]);
  }
  /*
  */
  if (myid == Host_ID) printf("\nOutput: Currentdensity [a.u.]\n");
  TRAN_Current_Spin(CDensity);
  TRAN_Current_OutPutCube(CDensity, TRAN_OffDiagonalCurrent);
  TRAN_Current_OutputVector(CDensity, TRAN_OffDiagonalCurrent);
  TRAN_Voronoi_CDEN(CDensity, TRAN_OffDiagonalCurrent);
  TRAN_Output_AveCdensity(CDensity);
  /*
  */
  for (ispin = 0; ispin <= SpinP_switch; ispin++){
    for (iaxis = 0; iaxis < 3; iaxis++){
      free(CDensity[ispin][iaxis]);
    }
    free(CDensity[ispin]);
  }
  free(CDensity);
  free(RRho);
  for (iside = 0; iside < 2; iside++){
    free(RJmat[iside]);
    free(Jbound[iside]);
  }
  free(RJmat);
  free(Jbound);
}
